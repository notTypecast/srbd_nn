#ifndef EPISODE_HPP
#define EPISODE_HPP
#include <Eigen/Core>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "src/opt/Individual.hpp"
#include "src/params.hpp"

#include "src/sim/single_rigid_body_dynamics.hpp"

#include "algevo/src/algevo/algo/cem.hpp"

#include "robot_dart/gui/magnum/graphics.hpp"
#include "robot_dart/robot_dart_simu.hpp"

using Algo = algevo::algo::CrossEntropyMethod<pq::opt::ControlIndividual>;

namespace pq {
    namespace train {
        class Episode {
        public:
            Episode()
            {
                _train_input = Eigen::MatrixXd(8, pq::Value::Param::Train::collection_steps);
                _train_target = Eigen::MatrixXd(3, pq::Value::Param::Train::collection_steps);
                _params.dim = pq::opt::ControlIndividual::dim;
                _params.pop_size = pq::Value::Param::Opt::pop_size;
                _params.num_elites = pq::Value::Param::Opt::num_elites;
                _params.max_value
                    = Algo::x_t::Constant(_params.dim, pq::Value::Param::Opt::max_value);
                _params.min_value = Algo::x_t(_params.dim);
                for (int i = 0; i < _params.dim; i += 3) {
                    _params.min_value.segment(i, 3) = pq::Value::Param::Opt::min_value;
                }
                _params.init_std
                    = Algo::x_t::Constant(_params.dim, pq::Value::Param::Opt::init_std);
            }

            std::vector<double> run(robot_dart::RobotDARTSimu& simu,
                std::shared_ptr<robot_dart::Robot>& body,
                std::shared_ptr<robot_dart::Robot>& fr_foot,
                std::shared_ptr<robot_dart::Robot>& fl_foot,
                std::shared_ptr<robot_dart::Robot>& rr_foot,
                std::shared_ptr<robot_dart::Robot>& rl_foot)
            {
                _params.init_mu = Algo::x_t::Constant(_params.dim, pq::Value::Param::Opt::init_mu);

                srbd::SingleRigidBodyDynamics srbd_obj
                    = srbd::SingleRigidBodyDynamics::create_robot(srbd::RobotType::ANYmal);
                srbd_obj.set_dt(pq::Value::Param::Sim::dt);

                std::vector<double> errors(pq::Value::Param::Train::collection_steps, 0);

                for (int i = 0; i < pq::Value::Param::Train::collection_steps; ++i) {
                    pq::Value::set_init_state(srbd_obj.base_position(), srbd_obj.base_velocity(),
                        srbd_obj.base_orientation(), srbd_obj.base_angular_velocity(),
                        srbd_obj.feet_positions(), srbd_obj.feet_phases());

                    Algo cem(_params);

                    for (int j = 0; j < pq::Value::Param::Opt::steps; ++j) {
                        cem.step(true);
                    }
                    std::cout << "Best cost: " << cem.best_value() << std::endl;

                    _params.init_mu = cem.best();
                    Eigen::Vector<double, 12> controls = cem.best().segment(0, 12);
                    std::cout << "Controls: " << controls.transpose() << std::endl;
                    std::cout << "Acceleration caused: "
                              << pq::opt::dynamic_model_predict(pq::Value::init_base_position,
                                     pq::Value::init_base_orientation,
                                     pq::Value::init_base_angular_vel,
                                     pq::Value::init_feet_positions, pq::Value::init_feet_phases,
                                     {controls.segment(0, 3), controls.segment(3, 3),
                                         controls.segment(6, 3), controls.segment(9, 3)},
                                     true)
                                     .transpose()
                              << std::endl;

                    srbd_obj.integrate({controls.segment(0, 3), controls.segment(3, 3),
                        controls.segment(6, 3), controls.segment(9, 3)});

                    errors[i]
                        = (pq::Value::Param::Opt::target - srbd_obj.base_position()).squaredNorm();

                    /*
                    std::cout << "Acc: " << acc.transpose() << std::endl;
                    std::cout << "Last acc: " << srbd_obj._last_acc.transpose() << std::endl;
                    std::cout << (srbd_obj._last_acc - acc).transpose() << std::endl;

                    std::cout << errors[i] << std::endl;
                    */

                    if (simu.schedule(simu.control_freq())) {
                        // Need to set position based on srbd_obj state
                        // For anymal, this requires inverse kinematics
                        body->set_base_pose(srbd_obj.base_tf());

                        auto feet_positions = srbd_obj.feet_positions();
                        fr_foot->set_base_pose(
                            (Eigen::Vector<double, 6>() << 0, 0, 0, feet_positions[0]).finished());
                        fl_foot->set_base_pose(
                            (Eigen::Vector<double, 6>() << 0, 0, 0, feet_positions[1]).finished());
                        rr_foot->set_base_pose(
                            (Eigen::Vector<double, 6>() << 0, 0, 0, feet_positions[2]).finished());
                        rl_foot->set_base_pose(
                            (Eigen::Vector<double, 6>() << 0, 0, 0, feet_positions[3]).finished());
                    }

                    simu.step_world();

                    /*
                    _train_input.col(i)
                        = (Eigen::Vector<double, 8>() << pq::Value::init_state, controls)
                              .finished();
                    _train_target.col(i) = p.get_last_ddq()
                        - pq::opt::dynamic_model_predict(pq::Value::init_state, controls);
                    */
                }

                ++_episode;

                return errors;
            }

            Eigen::MatrixXd get_train_input() { return _train_input; }

            Eigen::MatrixXd get_train_target() { return _train_target; }

            void set_run(int run) { _run_iter = run; }

        private:
            Eigen::MatrixXd _train_input;
            Eigen::MatrixXd _train_target;
            Algo::Params _params;
            int _episode = 1;
            int _run_iter = -1;
        };
    } // namespace train
} // namespace pq

#endif
