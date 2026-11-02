#include "perception/perceptionNode.hpp"

#include <chrono>
#include <string>

namespace perception
{

PerceptionNode::PerceptionNode()
: Node("perception_node")
{
    sub_pointcloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "pointcloud",
        10,
        std::bind(&PerceptionNode::pointCloudCallback, this, std::placeholders::_1));

    pub_scenario_ = this->create_publisher<crp_msgs::msg::Scenario>(
        "scenario",
        10);

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&PerceptionNode::publishScenario, this));
}

void PerceptionNode::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    (void)msg;
}

void PerceptionNode::publishScenario()
{
}

}  // namespace perception

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<perception::PerceptionNode>());
    rclcpp::shutdown();
    return 0;
}
