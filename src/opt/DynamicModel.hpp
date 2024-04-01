#ifndef DYNAMIC_MODEL_HPP
#define DYNAMIC_MODEL_HPP
#include <Eigen/Core>

#include "src/params.hpp"

#include "src/sim/utils.hpp"

namespace pq {
    namespace opt {
        constexpr int n_feet = 4;

        std::pair<srbd::Matrix, srbd::Vec6d> inv_mass_matrix(const srbd::Vec3d& base_position,
            const srbd::RotMat& base_orientation, const srbd::Vec3d& base_angular_vel,
            const std::vector<srbd::Vec3d>& feet_positions)
        {
            srbd::Matrix M = srbd::Matrix::Zero(6, n_feet * 3);
            srbd::Vec6d v;

            for (size_t i = 0; i < n_feet; i++) {
                // Linear part
                M.block(0, i * 3, 3, 3)
                    = srbd::Matrix::Identity(3, 3) / pq::Value::Param::Opt::CEM::mass;
                // Angular part
                M.block(3, i * 3, 3, 3) = pq::Value::Param::Opt::CEM::inertia_inv
                    * base_orientation.transpose()
                    * srbd::skew(feet_positions.at(i) - base_position);
            }

            v.head(3) = pq::Value::Param::Opt::CEM::g_vec;
            v.tail(3) = -pq::Value::Param::Opt::CEM::inertia_inv * srbd::skew(base_angular_vel)
                * (pq::Value::Param::Opt::CEM::inertia * base_angular_vel);

            return {M, v};
        }

        srbd::Vec6d dynamic_model_predict(
            // State
            const srbd::Vec3d& base_position, const srbd::RotMat& base_orientation,
            const srbd::Vec3d& base_angular_vel, const std::vector<srbd::Vec3d>& feet_positions,
            const Eigen::Vector4i& feet_stances,
            // Control
            const std::vector<srbd::Vec3d>& feet_forces)
        {
            srbd::Matrix M;
            srbd::Vec6d v;
            std::tie(M, v) = inv_mass_matrix(
                base_position, base_orientation, base_angular_vel, feet_positions);

            srbd::Vector F = Eigen::Map<srbd::Vector>(
                const_cast<double*>(&feet_forces[0][0]), static_cast<int>(3 * n_feet));

            for (size_t i = 0; i < n_feet; i++) {
                if (feet_stances[i] == 0) {
                    F.segment(i * 3, 3).setZero();
                }
            }

            srbd::Vec6d acc = (M * F) + v;
            // acc.head(3) += external_force / pq::Value::Param::Opt::mass;

            return acc;
        }
    } // namespace opt
} // namespace pq

#endif
