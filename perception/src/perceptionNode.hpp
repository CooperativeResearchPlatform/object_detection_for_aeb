#ifndef PERCEPTION_NODE_HPP_
#define PERCEPTION_NODE_HPP_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <crp_msgs/msg/scenario.hpp>
#include <crp_msgs/msg/ego.hpp>
#include <autoware_perception_msgs/msg/predicted_object.hpp>

#include "perception/brake_decision.hpp"
#include "perception/centerpoint_detector.hpp"

namespace perception
{

class PerceptionNode : public rclcpp::Node
{
public:
    PerceptionNode();

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void egoCallback(const crp_msgs::msg::Ego::SharedPtr msg);
    void publishScenario();

    // PointCloud2 -> (a) modell-bemenet [x,y,z,i,t] modell-frame-ben, (b) xyz szenzor-frame-ben
    bool convertCloud(const sensor_msgs::msg::PointCloud2 & msg,
                      std::vector<float> & model_points,
                      std::vector<float> & sensor_xyz);
    // modell-frame (nuScenes LiDAR) -> szenzor-frame (REP-103)
    Detection3D toSensorFrame(const Detection3D & d) const;
    autoware_perception_msgs::msg::PredictedObject toPredictedObject(const Detection3D & d) const;
    void publishBrake(bool brake, float distance);

    // --- ROS interfészek ---
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pointcloud_;
    rclcpp::Subscription<crp_msgs::msg::Ego>::SharedPtr sub_ego_;
    rclcpp::Publisher<crp_msgs::msg::Scenario>::SharedPtr pub_scenario_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_brake_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_obstacle_distance_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::CallbackGroup::SharedPtr inference_group_;
    rclcpp::CallbackGroup::SharedPtr io_group_;

    // --- modell és döntés ---
    std::unique_ptr<CenterPointDetector> detector_;
    std::unique_ptr<BrakeDecision> brake_decision_;

    // --- paraméterek ---
    double input_yaw_rad_ = 0.0;     // szenzor -> modell frame forgatás (z körül)
    double input_z_offset_ = 0.0;    // szenzor -> modell frame z eltolás
    double intensity_scale_ = 1.0;
    std::string intensity_field_;
    double publish_score_threshold_ = 0.3;
    double cloud_timeout_s_ = 0.3;
    bool brake_on_timeout_ = true;
    double ego_timeout_s_ = 0.5;
    double fallback_ego_speed_ = 10.0;
    std::vector<std::string> static_classes_;

    // --- állapot ---
    std::mutex mutex_;
    std::vector<Detection3D> latest_detections_;   // szenzor frame-ben
    std_msgs::msg::Header latest_header_;
    rclcpp::Time last_cloud_time_{0, 0, RCL_ROS_TIME};
    bool have_cloud_ = false;

    std::mutex ego_mutex_;
    float ego_speed_ = 0.0f;
    rclcpp::Time last_ego_time_{0, 0, RCL_ROS_TIME};
    bool have_ego_ = false;

    std::vector<float> model_points_;
    std::vector<float> sensor_xyz_;
};

}  // namespace perception

#endif  // PERCEPTION_NODE_HPP_
