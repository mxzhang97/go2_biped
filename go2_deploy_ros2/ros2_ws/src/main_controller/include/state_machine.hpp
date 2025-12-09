#pragma once

#include "biped_controller.hpp"
#include "rclcpp/rclcpp.hpp"



class StateMachine
{
public:
    //state machine states
    enum class STATES
    {
        DAMP = 0,
        CROUCH = 1,
        STAND =2,
        POLICY=3
    };

    STATES state = STATES::DAMP;

    bool Damp()
    {
        state = STATES::DAMP;
        return true;
    }

    bool Crouch()
    {
        if(state == STATES::DAMP)
        {
            state = STATES::CROUCH;
            crouch_count = 0;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool Stand()
    {
        if (state == STATES::CROUCH)
        {
            state = STATES::STAND;
            stand_count = 0;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool Policy()
    {
        if (state == STATES::STAND)
        {
            state = STATES::POLICY;
            return true;
        }
        else
        {
            return false;
        }
    }


    void Crouching(BipedController &biped_controller)
    {
        crouching_percentage = (float) crouch_count/crouch_duration;

        for (int i = 0; i<12;i++)
        {
            biped_controller.jpos_des[i] = (1-crouching_percentage) * biped_controller.start_pos[i] + crouching_percentage * biped_controller.sit_pos[i];
        }
        crouch_count++;
        crouch_count = crouch_count > crouch_duration ? crouch_duration : crouch_count;
    }

    void Standing(BipedController &biped_controller)
    {
        standing_percentage = (float) stand_count/stand_duration;
        for (int i=0; i<12; i++)
        {
            biped_controller.jpos_des[i] = (1- standing_percentage) * biped_controller.start_pos[i] + standing_percentage * biped_controller.stand_pos[i];
        }
        stand_count++;
        stand_count = stand_count > stand_duration ? stand_duration: stand_count;
    }

    bool CheckSafetyBoundaries(RobotStateInterface &robot_state_interface, const rclcpp::Logger &logger)
    {


        // Body/Hip: -48 to 48 deg -> [-0.838, 0.838]
        // With margin: [-0.788, 0.788]
        const float HIP_MIN = -0.788f*3;  //3 is good?
        const float HIP_MAX =  0.788f*3; 
        
        // Thigh: -200 to 90 deg -> [-3.491, 1.571]
        // With margin: [-3.441, 1.521]
        const float THIGH_MIN = -3.441f*2; 
        const float THIGH_MAX =  3.441f*2; 
        
        // Calf: -156 to -48 deg -> [-2.723, -0.838]
        // With margin: [-2.673, -0.788]
        const float CALF_MIN = -2.673f*2; 
        const float CALF_MAX = 2.f*2; 


        for(int i =0; i<4; i++)
        {
            int index = i*3;
            float q_hip_des = robot_state_interface.jpos_des[index];
            float q_thigh_des = robot_state_interface.jpos_des[index+1];
            float q_calf_des = robot_state_interface.jpos_des[index+2];

            if(q_hip_des < HIP_MIN || q_hip_des > HIP_MAX){
            RCLCPP_WARN(logger, "Leg %d hip joint target limits violated, Val: %.3f, Limit: %.3f, %.3f", i, q_hip_des, HIP_MIN, HIP_MAX);
                return true;
            }
            if(q_thigh_des < THIGH_MIN || q_thigh_des > THIGH_MAX){
            RCLCPP_WARN(logger, "Leg %d thigh joint target limits violated, Val: %.3f, Limit: %.3f, %.3f", i, q_thigh_des, THIGH_MIN, THIGH_MAX);
                return true;
            }
            if(q_calf_des < CALF_MIN || q_calf_des > CALF_MAX){
            RCLCPP_WARN(logger, "Leg %d calf joint target limits violated, Val: %.3f, Limit: %.3f, %.3f", i, q_calf_des, CALF_MIN, CALF_MAX);
                return true;
            }
        }
        if(robot_state_interface.projected_gravity[2] >0.6){
            return true;
        }
        return false;
    }


private:
    int stand_count, crouch_count;
    float crouching_percentage, standing_percentage;
    int stand_duration=100, crouch_duration=100;
};