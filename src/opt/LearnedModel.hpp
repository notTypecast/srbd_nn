#ifndef LEARNED_MODEL_HPP
#define LEARNED_MODEL_HPP
#include <Eigen/Core>
#include <vector>

#include "simple_nn/src/simple_nn/activation.hpp"
#include "simple_nn/src/simple_nn/layer.hpp"
#include "simple_nn/src/simple_nn/loss.hpp"
#include "simple_nn/src/simple_nn/neural_net.hpp"
#include "simple_nn/src/simple_nn/opt.hpp"

#include "src/params.hpp"

namespace pq {
    namespace opt {
        class NNModel {
        public:
            NNModel(const std::vector<int>& hidden_layers, int input_size, int output_size)
                : _input_size(input_size), _output_size(output_size)
            {
                assert(hidden_layers.size() > 0 && "expected at least one hidden layer");
                assert(input_size > 0 && output_size > 0
                    && "expected input and output size to be positive");
                _network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::ReLU>>(
                    input_size, hidden_layers[0]);
                for (ulong i = 1; i < hidden_layers.size(); ++i) {
                    _network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::ReLU>>(
                        hidden_layers[i - 1], hidden_layers[i]);
                }
                _network.add_layer<simple_nn::FullyConnectedLayer<simple_nn::Linear>>(
                    hidden_layers.back(), output_size);

                _network.set_weights(Eigen::VectorXd::Zero(_network.num_weights()));
                _optimizer.reset(Eigen::VectorXd::Zero(_network.num_weights()));
            }

            void train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target)
            {
                assert(input.rows() == _input_size && target.rows() == _output_size
                    && "incorrect input or target size");

                auto eval = [&](const Eigen::VectorXd& params) {
                    _network.set_weights(params);
                    Eigen::VectorXd dtheta
                        = _network.backward<simple_nn::MeanSquaredError>(input, target);
                    return std::make_pair(0.0, dtheta);
                };

                Eigen::VectorXd theta;

                for (int i = 0; i < pq::Value::Param::NN::epochs; ++i) {
                    bool stop;
                    std::tie(stop, std::ignore, theta) = _optimizer.optimize_once(eval);
                    _network.set_weights(theta);

                    if (stop) {
                        break;
                    }
                }
                _trained = true;
            }

            Eigen::VectorXd predict(const Eigen::VectorXd& input)
            {
                assert(input.size() == _input_size && "incorrect input size");
                return _network.forward(input);
            }

            void reset()
            {
                _trained = false;
                _network.set_weights(Eigen::VectorXd::Zero(_network.num_weights()));
                _optimizer.reset(Eigen::VectorXd::Zero(_network.num_weights()));
            }

            bool trained() { return _trained; }

        private:
            simple_nn::NeuralNet _network;
            simple_nn::Adam _optimizer;
            const int _input_size, _output_size;
            bool _trained = false;
        };

        Eigen::VectorXd convert_to_nn_input(const srbd::Vec3d& base_position,
            const srbd::RotMat& base_orientation, const srbd::Vec3d& base_angular_vel,
            const std::vector<srbd::Vec3d>& feet_positions, const std::vector<size_t>& feet_phases,
            const std::vector<srbd::Vec3d>& feet_forces)
        {
            Eigen::VectorXd input(pq::Value::Param::NN::input_size);
            input << base_position, base_orientation.block<3, 1>(0, 0),
                base_orientation.block<3, 1>(0, 1), base_orientation.block<3, 1>(0, 2),
                base_angular_vel,
                Eigen::Map<Eigen::VectorXd>(const_cast<double*>(&feet_positions[0][0]), 12), 0, 0,
                0, 0, Eigen::Map<Eigen::VectorXd>(const_cast<double*>(&feet_forces[0][0]), 12);

            for (size_t j = 0; j < pq::Value::init_feet_phases.size(); ++j) {
                input(27 + j) = pq::Value::init_feet_phases[j];
            }

            return input;
        }
    } // namespace opt
} // namespace pq

#endif
