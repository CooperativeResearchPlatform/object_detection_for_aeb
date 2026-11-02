#ifndef PERCEPTION_NODE_HPP_
#define PERCEPTION_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <crp_msgs/msg/scenario.hpp>

namespace perception
{

class PerceptionNode : public rclcpp::Node
{
public:
    PerceptionNode();

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void publishScenario();

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pointcloud_;
    rclcpp::Publisher<crp_msgs::msg::Scenario>::SharedPtr pub_scenario_;
    rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace perception

#endif  // PERCEPTION_NODE_HPP_
