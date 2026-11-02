#include "perception/perceptionNode.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/shape.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace perception
{

namespace
{
using sensor_msgs::msg::PointField;
using autoware_perception_msgs::msg::ObjectClassification;

struct FieldInfo
{
    int offset = -1;
    uint8_t datatype = 0;
};

FieldInfo findField(const sensor_msgs::msg::PointCloud2 & msg, const std::string & name)
{
    for (const auto & f : msg.fields) {
        if (f.name == name) {
            return {static_cast<int>(f.offset), f.datatype};
        }
    }
    return {};
}

float readAsFloat(const uint8_t * p, uint8_t datatype)
{
    switch (datatype) {
        case PointField::FLOAT32: { float v; std::memcpy(&v, p, 4); return v; }
        case PointField::FLOAT64: { double v; std::memcpy(&v, p, 8); return static_cast<float>(v); }
        case PointField::INT8:    { int8_t v; std::memcpy(&v, p, 1); return v; }
        case PointField::UINT8:   { uint8_t v; std::memcpy(&v, p, 1); return v; }
        case PointField::INT16:   { int16_t v; std::memcpy(&v, p, 2); return v; }
        case PointField::UINT16:  { uint16_t v; std::memcpy(&v, p, 2); return v; }
        case PointField::INT32:   { int32_t v; std::memcpy(&v, p, 4); return static_cast<float>(v); }
        case PointField::UINT32:  { uint32_t v; std::memcpy(&v, p, 4); return static_cast<float>(v); }
        default: return std::numeric_limits<float>::quiet_NaN();
    }
}

float normalizeAngle(float a)
{
    return std::atan2(std::sin(a), std::cos(a));
}

// mmdet3d nuScenes osztály -> Autoware ObjectClassification
uint8_t toAutowareLabel(const std::string & cls)
{
    if (cls == "car") return ObjectClassification::CAR;
    if (cls == "truck" || cls == "construction_vehicle") return ObjectClassification::TRUCK;
    if (cls == "bus") return ObjectClassification::BUS;
    if (cls == "trailer") return ObjectClassification::TRAILER;
    if (cls == "motorcycle") return ObjectClassification::MOTORCYCLE;
    if (cls == "bicycle") return ObjectClassification::BICYCLE;
    if (cls == "pedestrian") return ObjectClassification::PEDESTRIAN;
    return ObjectClassification::UNKNOWN;  // barrier, traffic_cone
}
}  // namespace

PerceptionNode::PerceptionNode()
: Node("perception_node")
{
    // ---------------- paraméterek ----------------
    CenterPointConfig det_cfg;
    det_cfg.pfe_model_path = declare_parameter<std::string>("model.pfe_onnx", "");
    det_cfg.head_model_path = declare_parameter<std::string>("model.head_onnx", "");
    det_cfg.use_cuda = declare_parameter<bool>("model.use_cuda", true);
    det_cfg.intra_op_threads = declare_parameter<int>("model.cpu_threads", 4);
    det_cfg.score_threshold = static_cast<float>(declare_parameter<double>("model.decode_score_threshold", 0.1));

    // A modell nuScenes LIDAR_TOP frame-ben tanult (x jobbra, y előre). Egy REP-103
    // szerinti (x előre, y balra) szenzornál ezért +90 fokkal kell forgatni.
    input_yaw_rad_ = declare_parameter<double>("input.yaw_rotation_deg", 90.0) * M_PI / 180.0;
    // nuScenes-ben a LiDAR ~1.84 m magasan van; eltérő szerelésnél ezzel lehet kiegyenlíteni.
    input_z_offset_ = declare_parameter<double>("input.z_offset_m", 0.0);
    intensity_field_ = declare_parameter<std::string>("input.intensity_field", "intensity");
    // nuScenes intenzitás 0..255; ha a szenzor 0..1-et ad, ide 255 kell.
    intensity_scale_ = declare_parameter<double>("input.intensity_scale", 1.0);

    publish_score_threshold_ = declare_parameter<double>("output.score_threshold", 0.3);
    static_classes_ = declare_parameter<std::vector<std::string>>(
        "output.static_obstacle_classes", std::vector<std::string>{"barrier", "traffic_cone"});

    BrakeDecisionConfig bcfg;
    bcfg.front_offset_m = static_cast<float>(declare_parameter<double>("brake.front_offset_m", bcfg.front_offset_m));
    bcfg.half_width_m = static_cast<float>(declare_parameter<double>("brake.half_width_m", bcfg.half_width_m));
    bcfg.max_range_m = static_cast<float>(declare_parameter<double>("brake.max_range_m", bcfg.max_range_m));
    bcfg.min_score = static_cast<float>(declare_parameter<double>("brake.min_score", bcfg.min_score));
    bcfg.min_distance_m = static_cast<float>(declare_parameter<double>("brake.min_distance_m", bcfg.min_distance_m));
    bcfg.ttc_threshold_s = static_cast<float>(declare_parameter<double>("brake.ttc_threshold_s", bcfg.ttc_threshold_s));
    bcfg.max_comfort_decel = static_cast<float>(declare_parameter<double>("brake.max_comfort_decel", bcfg.max_comfort_decel));
    bcfg.use_object_velocity = declare_parameter<bool>("brake.use_object_velocity", bcfg.use_object_velocity);
    bcfg.geometric_check = declare_parameter<bool>("brake.geometric_check", bcfg.geometric_check);
    bcfg.sensor_height_m = static_cast<float>(declare_parameter<double>("brake.sensor_height_m", bcfg.sensor_height_m));
    bcfg.geometric_max_range_m = static_cast<float>(declare_parameter<double>("brake.geometric_max_range_m", bcfg.geometric_max_range_m));
    bcfg.min_obstacle_height_m = static_cast<float>(declare_parameter<double>("brake.min_obstacle_height_m", bcfg.min_obstacle_height_m));
    bcfg.max_obstacle_height_m = static_cast<float>(declare_parameter<double>("brake.max_obstacle_height_m", bcfg.max_obstacle_height_m));
    bcfg.min_obstacle_points = static_cast<int>(declare_parameter<int>("brake.min_obstacle_points", bcfg.min_obstacle_points));
    bcfg.frames_to_trigger = static_cast<int>(declare_parameter<int>("brake.frames_to_trigger", bcfg.frames_to_trigger));
    bcfg.frames_to_release = static_cast<int>(declare_parameter<int>("brake.frames_to_release", bcfg.frames_to_release));

    cloud_timeout_s_ = declare_parameter<double>("safety.pointcloud_timeout_s", 0.3);
    brake_on_timeout_ = declare_parameter<bool>("safety.brake_on_timeout", true);
    ego_timeout_s_ = declare_parameter<double>("safety.ego_timeout_s", 0.5);
    fallback_ego_speed_ = declare_parameter<double>("safety.fallback_ego_speed_mps", 10.0);

    // ---------------- modell betöltése ----------------
    if (det_cfg.pfe_model_path.empty() || det_cfg.head_model_path.empty()) {
        throw std::runtime_error("A model.pfe_onnx és model.head_onnx paramétereket meg kell adni");
    }
    detector_ = std::make_unique<CenterPointDetector>(det_cfg);
    brake_decision_ = std::make_unique<BrakeDecision>(bcfg);
    RCLCPP_INFO(get_logger(), "CenterPoint betöltve (%s)", detector_->usingCuda() ? "CUDA" : "CPU");

    // ---------------- ROS interfészek ----------------
    // Az inferencia külön callback groupban fut, így nem blokkolja a timert / ego-t.
    inference_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    io_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions inference_opts;
    inference_opts.callback_group = inference_group_;
    rclcpp::SubscriptionOptions io_opts;
    io_opts.callback_group = io_group_;

    // SensorDataQoS (best effort, depth 5): ha az inferencia lassabb, a régi felhőket eldobjuk.
    sub_pointcloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "pointcloud",
        rclcpp::SensorDataQoS(),
        std::bind(&PerceptionNode::pointCloudCallback, this, std::placeholders::_1),
        inference_opts);

    sub_ego_ = this->create_subscription<crp_msgs::msg::Ego>(
        "ego",
        10,
        std::bind(&PerceptionNode::egoCallback, this, std::placeholders::_1),
        io_opts);

    pub_scenario_ = this->create_publisher<crp_msgs::msg::Scenario>(
        "scenario",
        10);
    pub_brake_ = this->create_publisher<std_msgs::msg::Bool>("brake_request", 10);
    pub_obstacle_distance_ = this->create_publisher<std_msgs::msg::Float32>("obstacle_distance", 10);

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&PerceptionNode::publishScenario, this),
        io_group_);
}

void PerceptionNode::egoCallback(const crp_msgs::msg::Ego::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(ego_mutex_);
    ego_speed_ = static_cast<float>(msg->twist.twist.linear.x);
    last_ego_time_ = now();
    have_ego_ = true;
}

bool PerceptionNode::convertCloud(
    const sensor_msgs::msg::PointCloud2 & msg,
    std::vector<float> & model_points,
    std::vector<float> & sensor_xyz)
{
    const FieldInfo fx = findField(msg, "x");
    const FieldInfo fy = findField(msg, "y");
    const FieldInfo fz = findField(msg, "z");
    const FieldInfo fi = findField(msg, intensity_field_);
    if (fx.offset < 0 || fy.offset < 0 || fz.offset < 0) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "A pontfelhőben nincs x/y/z mező");
        return false;
    }
    if (fi.offset < 0) {
        RCLCPP_WARN_ONCE(get_logger(), "Nincs '%s' mező, az intenzitás 0 lesz (rontja a pontosságot)",
                         intensity_field_.c_str());
    }
    if (msg.is_bigendian) {
        RCLCPP_WARN_ONCE(get_logger(), "Big-endian pontfelhő: nem támogatott, az értékek hibásak lehetnek");
    }

    const size_t n = static_cast<size_t>(msg.width) * msg.height;
    const float c = static_cast<float>(std::cos(input_yaw_rad_));
    const float s = static_cast<float>(std::sin(input_yaw_rad_));
    const float zo = static_cast<float>(input_z_offset_);
    const float is = static_cast<float>(intensity_scale_);

    model_points.clear();
    sensor_xyz.clear();
    model_points.reserve(n * 5);
    sensor_xyz.reserve(n * 3);

    for (uint32_t row = 0; row < msg.height; ++row) {
        const uint8_t * row_ptr = msg.data.data() + static_cast<size_t>(row) * msg.row_step;
        for (uint32_t col = 0; col < msg.width; ++col) {
            const uint8_t * p = row_ptr + static_cast<size_t>(col) * msg.point_step;
            const float x = readAsFloat(p + fx.offset, fx.datatype);
            const float y = readAsFloat(p + fy.offset, fy.datatype);
            const float z = readAsFloat(p + fz.offset, fz.datatype);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                continue;
            }
            const float inten = fi.offset >= 0 ? readAsFloat(p + fi.offset, fi.datatype) * is : 0.0f;

            sensor_xyz.push_back(x);
            sensor_xyz.push_back(y);
            sensor_xyz.push_back(z);

            model_points.push_back(c * x - s * y);
            model_points.push_back(s * x + c * y);
            model_points.push_back(z + zo);
            model_points.push_back(std::isfinite(inten) ? inten : 0.0f);
            model_points.push_back(0.0f);  // time_lag: egyetlen sweep -> 0
        }
    }
    return true;
}

Detection3D PerceptionNode::toSensorFrame(const Detection3D & d) const
{
    const float c = static_cast<float>(std::cos(input_yaw_rad_));
    const float s = static_cast<float>(std::sin(input_yaw_rad_));
    Detection3D o = d;
    // inverz forgatás (-yaw)
    o.x = c * d.x + s * d.y;
    o.y = -s * d.x + c * d.y;
    o.z = d.z - static_cast<float>(input_z_offset_);
    o.vx = c * d.vx + s * d.vy;
    o.vy = -s * d.vx + c * d.vy;
    o.yaw = normalizeAngle(d.yaw - static_cast<float>(input_yaw_rad_));
    return o;
}

autoware_perception_msgs::msg::PredictedObject PerceptionNode::toPredictedObject(const Detection3D & d) const
{
    autoware_perception_msgs::msg::PredictedObject obj;
    const auto & names = centerPointClassNames();
    const std::string cls = (d.label >= 0 && d.label < static_cast<int>(names.size())) ? names[d.label] : "";

    obj.existence_probability = d.score;
    ObjectClassification oc;
    oc.label = toAutowareLabel(cls);
    oc.probability = d.score;
    obj.classification.push_back(oc);

    auto & pose = obj.kinematics.initial_pose_with_covariance.pose;
    pose.position.x = d.x;
    pose.position.y = d.y;
    pose.position.z = d.z;
    pose.orientation.z = std::sin(d.yaw * 0.5);
    pose.orientation.w = std::cos(d.yaw * 0.5);

    // Autoware konvenció: a twist az objektum saját (heading) frame-jében van.
    const float c = std::cos(d.yaw), s = std::sin(d.yaw);
    auto & twist = obj.kinematics.initial_twist_with_covariance.twist;
    twist.linear.x = c * d.vx + s * d.vy;
    twist.linear.y = -s * d.vx + c * d.vy;

    obj.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
    obj.shape.dimensions.x = d.length;
    obj.shape.dimensions.y = d.width;
    obj.shape.dimensions.z = d.height;
    // object_id: tracking nélkül nincs stabil azonosító, nullán marad.
    return obj;
}

void PerceptionNode::publishBrake(bool brake, float distance)
{
    std_msgs::msg::Bool b;
    b.data = brake;
    pub_brake_->publish(b);
    std_msgs::msg::Float32 d;
    d.data = distance;
    pub_obstacle_distance_->publish(d);
}

void PerceptionNode::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    if (!convertCloud(*msg, model_points_, sensor_xyz_)) {
        return;
    }

    std::vector<Detection3D> dets_model;
    try {
        dets_model = detector_->detect(model_points_.data(), model_points_.size() / 5);
    } catch (const std::exception & e) {
        // nem frissítjük a last_cloud_time_-ot -> a timer timeout-ként kezeli
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "Inferencia hiba: %s", e.what());
        return;
    }

    std::vector<Detection3D> dets;
    dets.reserve(dets_model.size());
    for (const auto & d : dets_model) {
        if (d.score >= publish_score_threshold_) {
            dets.push_back(toSensorFrame(d));
        }
    }

    float ego_speed = static_cast<float>(fallback_ego_speed_);
    {
        std::lock_guard<std::mutex> lock(ego_mutex_);
        if (have_ego_ && (now() - last_ego_time_).seconds() < ego_timeout_s_) {
            ego_speed = ego_speed_;
        } else {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "Nincs friss ego sebesség, a döntés %.1f m/s feltételezett sebességgel fut", fallback_ego_speed_);
        }
    }

    const auto assessment = brake_decision_->evaluate(
        dets, sensor_xyz_.data(), sensor_xyz_.size() / 3, 3, ego_speed);
    publishBrake(assessment.brake, assessment.distance_m);

    const auto & t = detector_->lastTiming();
    RCLCPP_DEBUG(get_logger(),
        "pontok=%zu pillarek=%d | voxel %.1f ms, pfe %.1f ms, scatter %.1f ms, head %.1f ms, decode %.1f ms | det=%zu brake=%d",
        model_points_.size() / 5, t.num_voxels, t.voxelize_ms, t.pfe_ms, t.scatter_ms, t.head_ms, t.decode_ms,
        dets.size(), assessment.brake);

    std::lock_guard<std::mutex> lock(mutex_);
    latest_detections_ = std::move(dets);
    latest_header_ = msg->header;
    last_cloud_time_ = now();
    have_cloud_ = true;
}

void PerceptionNode::publishScenario()
{
    crp_msgs::msg::Scenario scenario;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool stale = !have_cloud_ || (now() - last_cloud_time_).seconds() > cloud_timeout_s_;
        if (stale) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Nincs friss detekció (> %.2f s)%s", cloud_timeout_s_,
                brake_on_timeout_ ? " -> biztonsági fékkérés" : "");
            if (brake_on_timeout_) {
                publishBrake(true, std::numeric_limits<float>::quiet_NaN());
            }
            return;
        }

        scenario.header = latest_header_;
        scenario.local_moving_objects.header = latest_header_;
        scenario.local_obstacles.header = latest_header_;
        const auto & names = centerPointClassNames();
        for (const auto & d : latest_detections_) {
            const std::string & cls = names[d.label];
            const bool is_static =
                std::find(static_classes_.begin(), static_classes_.end(), cls) != static_classes_.end();
            if (is_static) {
                scenario.local_obstacles.objects.push_back(toPredictedObject(d));
            } else {
                scenario.local_moving_objects.objects.push_back(toPredictedObject(d));
            }
        }
    }
    // paths és free_space: ezeket ez a node nem állítja elő, üresen maradnak.
    pub_scenario_->publish(scenario);
}

}  // namespace perception

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<perception::PerceptionNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
