#pragma once

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

namespace system_identification{

class SystemIdentification{

    public:
    // Løyser Ax = b for x
    Eigen::VectorXd solveleastSquares(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const{
        return A.colPivHouseholderQr().solve(b);
    }
};

}  // namespace system_identification

