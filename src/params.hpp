#ifndef PARAMS_HPP
#define PARAMS_HPP
#include <Eigen/Core>
#include <memory>

namespace pq {
    namespace opt {
        class NNModel;
    }
    namespace Value {
        // Constants / predefined values
        namespace Constant {
            constexpr double g = 9.81; // earth gravity
            constexpr double gm = 1.61; // moon gravity
            constexpr double mass = 30.4213964625; // default mass
        } // namespace Constant

        // Parameters for simulation and optimization
        namespace Param {
            namespace Sim {
                constexpr double dt = 0.01; // simulation time step
            } // namespace Sim
            namespace Opt {
                Eigen::Vector3d target = {1., 1., 0.2}; // target position for base
                Eigen::Matrix3d target_orientation
                    = Eigen::Matrix3d::Identity(); // target orientation for base
                Eigen::Vector3d g_vec = {0.0, 0.0, -Constant::g}; // gravity vector
                double mass = Constant::mass; // MPC mass
                Eigen::Matrix<double, 3, 3> inertia
                    = (Eigen::Matrix3d() << 0.88201174, -0.00137526, -0.00062895, -0.00137526,
                        1.85452968, -0.00018922, -0.00062895, -0.00018922, 1.97309185)
                          .finished(); // MPC inertia
                Eigen::Matrix<double, 3, 3> inertia_inv = inertia.inverse(); // inverse of inertia
                constexpr int steps = 80; // MPC steps
                constexpr int horizon = 80; // MPC horizon (number of control inputs per individual)
                constexpr int pop_size = 128; // population size
                constexpr int num_elites = 8; // number of elites
                constexpr double max_value
                    = Constant::mass * Constant::g; // maximum control input force
                Eigen::Vector3d min_value = (Eigen::Vector3d() << -max_value, -max_value, 0)
                                                .finished(); // minimum control input force
                constexpr double init_mu
                    = 0.25 * Constant::mass * Constant::g; // initial mean for CEM
                constexpr double init_std = 40; // initial standard deviation for CEM
            } // namespace Opt
            namespace NN {
                constexpr int epochs = 10000; // number of epochs for training
                constexpr int input_size
                    = 43; // input size for NN, TODO: see if base_vel should be included
                constexpr int output_size = 6; // output size for NN
            } // namespace NN
            namespace Train {
                constexpr int collection_steps
                    = 200; // number of steps to collect data for training (per episode)
                constexpr int episodes = 10; // number of episodes to train
                constexpr int runs = 1; // number of runs to train (for averaging)
            } // namespace Train
        } // namespace Param

        std::vector<Eigen::Vector<double, 3>> feet_ref_positions
            = {Eigen::Vector3d(0.34, 0.19, -0.42), Eigen::Vector3d(-0.34, 0.19, -0.42),
                Eigen::Vector3d(0.34, -0.19, -0.42), Eigen::Vector3d(-0.34, -0.19, -0.42)};
        constexpr size_t T = 40;
        constexpr size_t T_swing = 20;

        Eigen::Vector<double, 3> init_base_position;
        Eigen::Vector<double, 3> init_base_vel;
        Eigen::Matrix<double, 3, 3> init_base_orientation;
        Eigen::Vector<double, 3> init_base_angular_vel;
        std::vector<Eigen::Vector<double, 3>> init_feet_positions;
        std::vector<size_t> init_feet_phases;

        void set_init_state(Eigen::Vector<double, 3> base_position,
            Eigen::Vector<double, 3> base_vel, Eigen::Matrix<double, 3, 3> base_orientation,
            Eigen::Vector<double, 3> base_angular_vel,
            std::vector<Eigen::Vector<double, 3>> feet_positions, std::vector<size_t> feet_phases)
        {
            init_base_position = base_position;
            init_base_vel = base_vel;
            init_base_orientation = base_orientation;
            init_base_angular_vel = base_angular_vel;
            init_feet_positions = feet_positions;
            init_feet_phases = feet_phases;
        }

        std::unique_ptr<pq::opt::NNModel> learned_model;
    } // namespace Value
} // namespace pq

#endif