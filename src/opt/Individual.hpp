#ifndef INDIVIDUAL_HPP
#define INDIVIDUAL_HPP
#include <Eigen/Core>

#include "src/opt/DynamicModel.hpp"
#include "src/opt/LearnedModel.hpp"
#include "src/params.hpp"

#include "src/sim/utils.hpp"

#include <dart/math/Geometry.hpp>

namespace pq {
    namespace opt {
        void integrate_step(const srbd::Vec6d& acc, const double dt, srbd::Vec3d& base_position,
            srbd::Vec3d& base_vel, srbd::RotMat& base_orientation, srbd::Vec3d& base_angular_vel,
            std::vector<srbd::Vec3d>& feet_positions, std::vector<size_t>& feet_phases)
        {
            // (Semi-implicit) Euler integration
            base_vel += acc.head(3) * dt;
            base_position += base_vel * dt;

            base_angular_vel += acc.tail(3) * dt;
            base_orientation *= srbd::exp_angular(base_angular_vel * dt);

            // Increase phase variables
            double k_foot = 2. * (pq::Value::T - pq::Value::T_swing) * dt;
            for (size_t i = 0; i < n_feet; i++) {
                feet_phases[i]++;
                if (((feet_phases[i] % pq::Value::T) == 0) && pq::Value::T_swing > 0) {
                    feet_positions[i].head(2) = pq::Value::feet_ref_positions[i].head(2)
                        + k_foot * (base_orientation.transpose() * base_vel).head(2);
                    feet_positions[i][2] = 0.;

                    feet_positions[i] = base_position + base_orientation * feet_positions[i];
                    feet_positions[i][2] = 0; // TODO: this works only for flat terrain (z = 0)
                    // feet_positions[i][2] = _terrain->height(feet_positions[i][0],
                    // feet_positions[i][1]); // get height from terrain
                }
            }
        }

        struct ControlIndividual {
            static constexpr unsigned int dim = 12 * pq::Value::Param::Opt::horizon;

            // Individual: [f11', f12', f13', f14', ..., fh1', fh2', fh3', fh4']  (size 3*4h = 12h)
            // where fij' is the force applied to the j-th foot at the i-th time step (size 3)
            double eval(const Eigen::Matrix<double, 1, dim>& x)
            {
                srbd::Vec3d base_position = pq::Value::init_base_position;
                srbd::Vec3d base_vel = pq::Value::init_base_vel;
                srbd::RotMat base_orientation = pq::Value::init_base_orientation;
                srbd::Vec3d base_angular_vel = pq::Value::init_base_angular_vel;
                std::vector<srbd::Vec3d> feet_positions = pq::Value::init_feet_positions;
                std::vector<size_t> feet_phases = pq::Value::init_feet_phases;

                double cost = 0;

                for (ulong i = 0; i < dim; i += 12) {
                    std::vector<srbd::Vec3d> feet_forces;
                    for (int j = 0; j < n_feet; ++j) {
                        feet_forces.push_back(x.block<1, 3>(0, i + 3 * j).transpose());
                    }

                    srbd::Vec6d acc
                        = pq::opt::dynamic_model_predict(base_position, base_orientation,
                            base_angular_vel, feet_positions, feet_phases, feet_forces);
                    integrate_step(acc, pq::Value::Param::Sim::dt, base_position, base_vel,
                        base_orientation, base_angular_vel, feet_positions, feet_phases);

                    // cost += acc.norm();

                    cost += 5 * (pq::Value::Param::Opt::target - base_position).norm()
                        + dart::math::logMap(pq::Value::Param::Opt::target_orientation
                            * base_orientation.transpose())
                              .norm()
                        + base_vel.norm() + base_angular_vel.norm();
                }

                return -cost;
            }
        };
    } // namespace opt
} // namespace pq

#endif
