#include "system_identification/system_identification.hpp"
#include <rclcpp/rclcpp.hpp>

namespace system_identification{



class SystemIdentificationNode : public rclcpp::Node{

    public:
        explicit SystemIdentificationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
        
    private:
        SystemIdentification identifier_;
        rclcpp::TimerBase::SharedPtr timer_;


        void timerCallback(){
            Eigen::MatrixXd A(4, 2);
            A << 1.0, 0.0,
                 0.0, 1.0,
                 1.0, 1.0,
                 2.0, 0.0;

            Eigen::VectorXd b(4);
            b << 1.0,
                 2.0,
                 3.0,
                 4.0;

            Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

            RCLCPP_INFO(this->get_logger(), "x = [%f, %f]", x(0), x(1));
        }


};
SystemIdentificationNode::SystemIdentificationNode(const rclcpp::NodeOptions & options): rclcpp::Node("system_identification_node", options){

    RCLCPP_INFO(this->get_logger(), "System identification node started");

    timer_ = this->create_wall_timer(
        std::chrono::seconds(1),
        std::bind(&SystemIdentificationNode::timerCallback, this));
};
}  // namespace system_identification




int main(int argc, char ** argv){
    
  rclcpp::init(argc, argv);
  auto node = std::make_shared<system_identification::SystemIdentificationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

