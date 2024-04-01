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

#include "genetic_alg/src/GeneticAlg.hpp"

#include "robot_dart/gui/magnum/graphics.hpp"
#include "robot_dart/robot_dart_simu.hpp"

using Algo = algevo::algo::CrossEntropyMethod<pq::opt::ControlIndividual>;

namespace pq {
    namespace train {
        class Episode {
        public:
            Episode()
            {
                _train_input = Eigen::MatrixXd(
                    pq::Value::Param::NN::input_size, pq::Value::Param::Train::collection_steps);
                _train_target = Eigen::MatrixXd(
                    pq::Value::Param::NN::output_size, pq::Value::Param::Train::collection_steps);
                // CEM parameters
                _params_cem.dim = pq::opt::ControlIndividual::dim;
                _params_cem.pop_size = pq::Value::Param::Opt::CEM::pop_size;
                _params_cem.num_elites = pq::Value::Param::Opt::CEM::num_elites;
                _params_cem.max_value
                    = Algo::x_t::Constant(_params_cem.dim, pq::Value::Param::Opt::CEM::max_value);
                _params_cem.min_value = Algo::x_t(_params_cem.dim);
                for (size_t i = 0; i < _params_cem.dim; i += 3) {
                    _params_cem.min_value.segment(i, 3) = pq::Value::Param::Opt::CEM::min_value;
                }
                _params_cem.init_mu
                    = Algo::x_t::Constant(_params_cem.dim, pq::Value::Param::Opt::CEM::init_mu);
                _params_cem.init_std
                    = Algo::x_t::Constant(_params_cem.dim, pq::Value::Param::Opt::CEM::init_std);

                // GA parameters
                _params_ga.pop_size = pq::Value::Param::Opt::GA::pop_size;
                // Individual: [b11, b12, b13, b14, ..., bh1, bh2, bh3, bh4]  (size 4h)
                // each bij is 0 or 1, where 0 means foot j in swing phase and 1 means foot in
                // stance phase during i-th time step
                _params_ga.ind_size = pq::Value::Param::Opt::GA::horizon * 4;
                _params_ga.fitness_function = [&](const genetic_alg::Individual& ind) {
                    // TODO: Need way to pass GA individual to each CEM individual
                    Algo cem(_params_cem);
                    pq::Value::Param::Opt::CEM::tmp_curr_ga_ind
                        = ind; // TODO: cannot do it like this normally

                    for (int j = 0; j < pq::Value::Param::Opt::CEM::steps; ++j) {
                        cem.step(true);
                    }

                    if (pq::Value::Param::Opt::CEM::tmp_best_cem_score < cem.best_value()) {
                        pq::Value::Param::Opt::CEM::tmp_best_cem_score = cem.best_value();
                        pq::Value::Param::Opt::CEM::tmp_best_cem_ind = cem.best();
                        pq::Value::Param::Opt::CEM::tmp_best_ga_ind = ind;
                    }

                    return cem.best_value();
                };
            }

            std::vector<double> run(robot_dart::RobotDARTSimu& simu,
                std::shared_ptr<robot_dart::Robot>& body,
                std::shared_ptr<robot_dart::Robot>& fr_foot,
                std::shared_ptr<robot_dart::Robot>& fl_foot,
                std::shared_ptr<robot_dart::Robot>& rr_foot,
                std::shared_ptr<robot_dart::Robot>& rl_foot)
            {
                srbd::SingleRigidBodyDynamics srbd_obj
                    = srbd::SingleRigidBodyDynamics::create_robot(srbd::RobotType::ANYmal);
                srbd_obj.set_phase_handler<srbd::PredefinedPhaseHandler>();
                srbd_obj.set_dt(pq::Value::Param::Sim::dt);

                std::vector<double> errors(pq::Value::Param::Train::collection_steps, 0);

                for (int i = 0; i < pq::Value::Param::Train::collection_steps; ++i) {
                    pq::Value::set_init_state(srbd_obj.base_position(), srbd_obj.base_velocity(),
                        srbd_obj.base_orientation(), srbd_obj.base_angular_velocity(),
                        srbd_obj.feet_positions(), srbd_obj.feet_phases());

                    genetic_alg::GeneticAlg ga(_params_ga);

                    pq::Value::Param::Opt::CEM::tmp_best_cem_score = -std::numeric_limits<double>::max();

                    for (int j = 0; j < pq::Value::Param::Opt::GA::steps; ++j) {
                        std::cout << "GA step: " << j << std::endl;
                        ga.run_epoch();
                    }

                    // TODO: Need way to get best cem individual from ga individual

                    std::vector<bool> feet_phases(4);
                    for (int j = 0; j < 4; ++j) {
                        feet_phases[j] = pq::Value::Param::Opt::CEM::tmp_best_ga_ind[j];
                    }

                    Eigen::Vector<double, 12> controls = pq::Value::Param::Opt::CEM::tmp_best_cem_ind.segment(0, 12);
                    std::vector<Eigen::Vector3d> controls_vec = {controls.segment(0, 3),
                        controls.segment(3, 3), controls.segment(6, 3), controls.segment(9, 3)};

                    std::cout << "Controls: " << controls.transpose() << std::endl;

                    srbd_obj.integrate(controls_vec, feet_phases);

                    errors[i] = (pq::Value::Param::Opt::CEM::target - srbd_obj.base_position())
                                    .squaredNorm();

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
                    _train_input.col(i) = pq::opt::convert_to_nn_input(srbd_obj.base_position(),
                        srbd_obj.base_orientation(), srbd_obj.base_angular_velocity(),
                        srbd_obj.feet_positions(), srbd_obj.feet_phases(), controls_vec);

                    _train_target.col(i) = srbd_obj.last_acc()
                        - pq::opt::dynamic_model_predict(pq::Value::init_base_position,
                            pq::Value::init_base_orientation, pq::Value::init_base_angular_vel,
                            pq::Value::init_feet_positions, pq::Value::init_feet_phases,
                            controls_vec);
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
            Algo::Params _params_cem;
            genetic_alg::Parameters _params_ga;
            int _episode = 1;
            int _run_iter = -1;
        };
    } // namespace train
} // namespace pq

#endif
