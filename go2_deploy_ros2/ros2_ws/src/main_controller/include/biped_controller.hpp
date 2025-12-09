#pragma once

#include <array>
#include <vector>
#include <deque>

#include <iostream>
#include <filesystem>

#include "torch/script.h"
#include "robot_state_interface.hpp"
#include "gamepad.hpp"

#include "common/common_types.hpp"


namespace fs = std::filesystem;


class BipedController
{
public:
    BipedController()
    {

    }

    void load_model()
    {
        dt = 0.02;
        stand_kp = 60.0;
        stand_kd = 0.5;
        ctrl_kp = 25.0;
        ctrl_kd = 0.5;
        action_scale = 0.25;

        policy_name = "scripted_student_policy_cpu.pt"; 
        se_name = "scripted_state_estimator_cpu.pt"; 
        
        history_window_size = 5; 
        policy_obs_dim = 54;     
        se_obs_dim = 42; 


        unitree_indices_for_isaac_joints = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
        //unitree_indices_for_isaac_joints = {0,1,2,3,4,5,6,7,8,9,10,11};


        stand_pos = {-0.1f, 0.8f, -1.5f, 0.1f, 0.8f, -1.5f, -0.1f, 1.0f, -1.5f, 0.1f, 1.0f, -1.5f};
        sit_pos = {0.0f, 1.3f, -2.5f, 0.0f, 1.3f, -2.5f, 0.0f, 1.3f, -2.5f, 0.0f, 1.3f, -2.5f};


        std::vector<float> zero_se_obs(se_obs_dim, 0.0f);
        history_obs.assign(history_window_size, zero_se_obs);


        fs::path model_path = "/home/max/go2_biped/go2_deploy_ros2/ros2_ws/src/main_controller/models/";
        policy = torch::jit::load(model_path / policy_name);
        se = torch::jit::load(model_path / se_name);


        policy.eval();
        se.eval();


        std::cout << "Successfully loaded policy from: " << model_path / policy_name << std::endl;
        std::cout << "Successfully loaded State Estimator from: " << model_path / se_name << std::endl;
    }

    void reset()
    {
        base_lin_vel.fill(0.0f);
        com_cop.fill(0.0f);
        actions.fill(0.0f);
        vel_commands.fill(0.0f);
        arm_commands.fill(0.0f);
        quad_state = 1.0f;
        hri_mode = 0.0f;
    }

    void get_input(const RobotStateInterface &robot_interface, const GamepadInterface &gamepad, const hri_arm_targets &hri_arm_targets)
    {
        std::vector<float> jpos_unitree_delta(12);
        std::vector<float> jvel_unitree(12);

        for (int i = 0; i < 12; ++i)
        {
            //i is policy index
            int t = unitree_indices_for_isaac_joints[i];

            jpos_processed.at(t) = robot_interface.jpos.at(i) - stand_pos.at(i);
            jvel.at(t) = robot_interface.jvel.at(i);
        }
        
        std::copy(robot_interface.gyro.begin(), robot_interface.gyro.end(), base_ang_vel.begin());
        std::copy(robot_interface.projected_gravity.begin(), robot_interface.projected_gravity.end(), projected_gravity.begin());
        
        std::vector<float> current_se_obs(se_obs_dim);
        for(int i = 0; i < 3; ++i)  current_se_obs.at(i)    = base_ang_vel.at(i) * 0.25;
        for(int i = 0; i < 3; ++i)  current_se_obs.at(i+3)  = projected_gravity.at(i);
        for(int i = 0; i < 12; ++i) current_se_obs.at(i+6)  = jpos_processed.at(i); // Use absolute Isaac pos for SE
        for(int i = 0; i < 12; ++i) current_se_obs.at(i+18) = jvel.at(i) * 0.05;
        for(int i = 0; i < 12; ++i) current_se_obs.at(i+30) = actions.at(i);
        history_obs.pop_front();
        history_obs.push_back(current_se_obs);


        //toggle quad mode
        if (gamepad.R1.on_press) {
            quad_state = 1.0f - quad_state; 
            if (quad_state < 0.5f) {
                std::cout << "Biped Mode Active" << std::endl;
            } else { 
                std::cout << "Quad Mode Active" << std::endl;
                arm_commands.fill(0.2f);
                vel_commands.fill(0.0f);
                hri_mode = 0.0f;
            }
        }
        if(gamepad.Y.on_press){
            hri_mode = 1.0f - hri_mode;
            if(hri_mode < 0.5f){
                std::cout <<"Controller Mode" << std::endl;
                arm_commands.fill(0.2f);
            }
            else{
                std::cout<<"HRI Mode"<<std::endl;
                arm_commands.fill(0.2f);
            }
        }
        if(!hri_mode){
            if(gamepad.A.pressed)
            {
                arm_commands.at(0) = 0.6;
            }
            else{
                arm_commands.at(0) = 0.0;
            }
            if(gamepad.B.pressed)
            {
                arm_commands.at(1) = 0.6;
            }
            else 
            {
                arm_commands.at(1) = 0.0;
            }
        }
        else{
            arm_commands.at(0) = hri_arm_targets.arm_target_l.load();
            arm_commands.at(1) = hri_arm_targets.arm_target_r.load();
        }
        // record command
        vel_commands.at(1) = gamepad.ly; // linear_x: [-1,1]
        vel_commands.at(0) = -gamepad.lx; // linear_y; [-1,1]
        vel_commands.at(2) = -gamepad.rx; // angular_z: [-1,1]

    }

    void warmup()
    {
        torch::Tensor zero_se_input = torch::zeros({1, history_window_size * se_obs_dim});
        torch::Tensor zero_policy_input = torch::zeros({1, policy_obs_dim});
        se.forward({zero_se_input});
        policy.forward({zero_policy_input});
    }

    void forward()
    {
        torch::Tensor se_input_tensor = torch::zeros({1, history_window_size * se_obs_dim});
        std::vector<torch::jit::IValue> se_input;
        for(int i = 0; i < history_window_size; ++i) {
            for(int j = 0; j < se_obs_dim; ++j) {
                se_input_tensor[0][i * se_obs_dim + j] = history_obs.at(i).at(j);
            }
        }

        se_input.push_back(se_input_tensor);
        
        torch::Tensor se_output_tensor = se.forward(se_input).toTensor();
        
        for(int i = 0; i < 3; ++i) {
            base_lin_vel.at(i) = se_output_tensor[0][i].item<float>();
        }
        for(int i = 0; i < 3; ++i) {
            com_cop.at(i) = se_output_tensor[0][i+3].item<float>();
        }


        torch::Tensor policy_input_tensor = torch::zeros({1, policy_obs_dim});
        std::vector<torch::jit::IValue> policy_input;
        
        for(int i = 0; i < 3; ++i)  policy_input_tensor[0][i]    = base_lin_vel.at(i) * 1.0;
        for(int i = 0; i < 3; ++i)  policy_input_tensor[0][i+3]  = base_ang_vel.at(i) * 0.25 ;
        for(int i = 0; i < 3; ++i)  policy_input_tensor[0][i+6]  = projected_gravity.at(i);
        for(int i = 0; i < 3; ++i)  policy_input_tensor[0][i+9]  = com_cop.at(i);
        for(int i = 0; i < 12; ++i) policy_input_tensor[0][i+12]  = jpos_processed.at(i) * 1.0;
        for(int i = 0; i < 12; ++i) policy_input_tensor[0][i+24] = jvel.at(i) * 0.05;
        for(int i = 0; i < 12; ++i) policy_input_tensor[0][i+36] = actions.at(i);
        for(int i = 0; i < 3; ++i)  policy_input_tensor[0][i+48] = vel_commands.at(i);
        for(int i = 0; i < 2; ++i)  policy_input_tensor[0][i+51] = arm_commands.at(i);
        policy_input_tensor[0][53] = quad_state;

        policy_input.push_back(policy_input_tensor);
        torch::Tensor policy_output_tensor = policy.forward(policy_input).toTensor();

        std::array<float, 12> actions_scaled;
        for(int i = 0; i < 12; ++i)
        {
            int t = unitree_indices_for_isaac_joints[i];
            actions.at(i) = policy_output_tensor[0][i].item<float>();
            actions_scaled.at(i) = actions.at(i) * action_scale;
            jpos_des.at(t) = stand_pos.at(t) + actions_scaled.at(i);
        }
    }


    void save_jpos(RobotStateInterface &robot_interface)
    {
        std::copy(robot_interface.jpos.begin(), robot_interface.jpos.end(), start_pos.begin());
    }

    //write actions out
    std::array<float, 12> jpos_des;
    std::array<float, 12> start_pos;
    std::array<float, 12> sit_pos;
    std::array<float, 12> stand_pos;

    float stand_kp = 60.0;
    float stand_kd = 0.5;

    float ctrl_kp = 25.0;
    float ctrl_kd = 0.5;

private:
    std::string policy_name, se_name;
    int history_window_size, se_obs_dim, policy_obs_dim;
    float action_scale;
    float dt;

    torch::jit::script::Module policy;
    torch::jit::script::Module se;

    std::array<float, 3> base_lin_vel; //estimated 
    std::array<float, 3> base_ang_vel;
    std::array<float, 3> projected_gravity;
    std::array<float, 3> com_cop; //estimated
    std::array<float, 12> jpos_processed;
    std::array<float, 12> jvel;
    std::array<float, 12> actions; // prev actions

    std::array<float, 3> vel_commands;
    std::array<float, 2> arm_commands;
    float quad_state;
    float hri_mode;

    std::array<int, 12> unitree_indices_for_isaac_joints;
    std::array<int, 12> isaac_indices_for_unitree_joints;

    std::deque<std::vector<float>> history_obs;

};
