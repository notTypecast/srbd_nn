#include <Eigen/Core>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "src/opt/Individual.hpp"
#include "src/opt/LearnedModel.hpp"
#include "src/params.hpp"
#include "src/train/Episode.hpp"

#include "src/sim/single_rigid_body_dynamics.hpp"
#include "src/sim/utils.hpp"

#include "robot_dart/gui/magnum/graphics.hpp"
#include "robot_dart/robot_dart_simu.hpp"

using Algo = algevo::algo::CrossEntropyMethod<pq::opt::ControlIndividual>;

int main(int argc, char** argv)
{
    // Create robotdart robot
    /*
    std::pair<std::string, std::string> workaround;
    workaround.first = "anymal_b_simple_description";
    workaround.second = "models/anymal_b_simple_description";
    std::vector<std::pair<std::string, std::string>> packages = {workaround};
    auto anymal_rd = std::make_shared<robot_dart::Robot>(
        "models/anymal_b_simple_description/anymal.urdf", packages, "anymal");
    anymal_rd->set_color_mode("material");
    */

    auto box = robot_dart::Robot::create_box(
        {0.2, 0.05, 0.04}, {0, 0, 0, 0, 0, 0}, "free", 1, dart::Color::Red(1.0), "body");
    auto fr_foot = robot_dart::Robot::create_ellipsoid(
        {0.02, 0.02, 0.02}, {0, 0, 0, 0, 0, 0}, "free", 1, dart::Color::Red(1.0), "fr_foot");
    auto fl_foot = robot_dart::Robot::create_ellipsoid(
        {0.02, 0.02, 0.02}, {0, 0, 0, 0, 0, 0}, "free", 1, dart::Color::Blue(1.0), "fl_foot");
    auto rr_foot = robot_dart::Robot::create_ellipsoid(
        {0.02, 0.02, 0.02}, {0, 0, 0, 0, 0, 0}, "free", 1, dart::Color::Red(0.5), "rr_foot");
    auto rl_foot = robot_dart::Robot::create_ellipsoid(
        {0.02, 0.02, 0.02}, {0, 0, 0, 0, 0, 0}, "free", 1, dart::Color::Blue(0.5), "rl_foot");

    auto target = robot_dart::Robot::create_ellipsoid({0.05, 0.05, 0.05},
        {0, 0, 0, pq::Value::Param::Opt::CEM::target[0], pq::Value::Param::Opt::CEM::target[1],
            pq::Value::Param::Opt::CEM::target[2]},
        "free", 1, dart::Color::Blue(1.0), "target");

    // Simulator
    robot_dart::RobotDARTSimu simu(pq::Value::Param::Sim::dt);
    simu.set_collision_detector("fcl");

    simu.add_visual_robot(box);
    simu.add_visual_robot(fr_foot);
    simu.add_visual_robot(fl_foot);
    simu.add_visual_robot(rr_foot);
    simu.add_visual_robot(rl_foot);
    simu.add_visual_robot(target);
    simu.add_checkerboard_floor();

    // Graphics
    auto graphics = std::make_shared<robot_dart::gui::magnum::Graphics>();
    simu.set_graphics(graphics);
    graphics->look_at({0., 2., 1.5}, {0., 0., 0.5});
    graphics->record_video("test.mp4");

    std::vector<std::vector<double>> errors_per_episode(
        pq::Value::Param::Train::runs * pq::Value::Param::Train::episodes);

    pq::Value::learned_model = std::make_unique<pq::opt::NNModel>(std::vector<int>{12, 6, 4},
        pq::Value::Param::NN::input_size, pq::Value::Param::NN::output_size);

    std::vector<double> gravity_values = {18, 36, 100};

    for (const double& g : gravity_values) {
        pq::Value::Param::Opt::CEM::g_vec[2] = -g;

        std::cout << "Running with g = " << -pq::Value::Param::Opt::CEM::g_vec[2] << std::endl;

        for (int j = 0; j < pq::Value::Param::Train::runs; ++j) {
            std::srand(std::time(NULL));
            std::cout << "Run " << j << std::endl;
            pq::Value::learned_model->reset();
            pq::train::Episode episode;
            episode.set_run(j + 1);

            int run_idx = j * pq::Value::Param::Train::episodes;
            std::cout << "Episode: ";
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k) {
                std::cout << k << " " << std::flush;
                errors_per_episode[run_idx + k]
                    = episode.run(simu, box, fr_foot, fl_foot, rr_foot, rl_foot);
                /*
                // Early stopping if the change in error is smaller than a threshold
                // This will not work if writing to a file
                if (k > 0 && errors_per_episode[run_idx + k -
                1][pq::Value::Param::Train::collection_steps - 1] - errors_per_episode[run_idx +
                k][pq::Value::Param::Train::collection_steps - 1] < 10e-4)
                {
                    std::cout << ":done";
                    break;
                }
                */

                pq::Value::learned_model->train(
                    episode.get_train_input(), episode.get_train_target());
            }
            std::cout << std::endl;
        }

        std::ofstream out(
            "sample_error/error_" + std::to_string(-pq::Value::Param::Opt::CEM::g_vec[2]) + ".txt");
        out << -pq::Value::Param::Opt::CEM::g_vec[2] << " " << pq::Value::Param::Train::collection_steps
            << " " << pq::Value::Param::Train::episodes << " " << pq::Value::Param::Train::runs
            << " " << std::endl;
        for (int j = 0; j < pq::Value::Param::Train::runs; ++j) {
            int run_idx = j * pq::Value::Param::Train::episodes;
            for (int k = 0; k < pq::Value::Param::Train::episodes; ++k) {
                for (int l = 0; l < pq::Value::Param::Train::collection_steps; ++l) {
                    out << errors_per_episode[run_idx + k][l] << " ";
                }
                out << std::endl;
            }
        }
        out.close();
    }

    return 0;
}