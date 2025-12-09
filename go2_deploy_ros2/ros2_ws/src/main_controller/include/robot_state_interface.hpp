#pragma once

#include <cmath>
#include <array>
#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/imu_state.hpp"
#include "unitree_go/msg/motor_state.hpp"



class RobotStateInterface
{
public:
    RobotStateInterface()
    {
        jpos_des.fill(0.);
        jvel_des.fill(0.);
        kp.fill(0.);
        kd.fill(0.);
        tau_ff.fill(0.);
        projected_gravity.fill(0.);
        projected_gravity.at(2) = -1.0;
    }
    
    void set(unitree_go::msg::LowState &state)
    {
        //imu
        const unitree_go::msg::IMUState &imu = state.imu_state;
        quat = imu.quaternion;
        rpy = imu.rpy;
        gyro = imu.gyroscope;
        acc = imu.accelerometer;
        UpdateProjectedGravity();

        //motor
        const std::array<unitree_go::msg::MotorState, 20> &motor = state.motor_state;
        for (size_t i =0; i<12; i++)
        {
            const unitree_go::msg::MotorState &m = motor[i];
            jpos[i] = m.q;
            jvel[i] = m.dq;
            tau[i] = m.tau_est;
        }
    }
    std::array<float, 12> jpos, jvel, tau;
    std::array<float, 4> quat;
    std::array<float, 3> rpy,gyro,projected_gravity,acc;
    std::array<float, 12> jpos_des, jvel_des, kp, kd, tau_ff;

private:
    inline void UpdateProjectedGravity()
    {
        // inverse quat
        float w = quat.at(0);
        float x = -quat.at(1);
        float y = -quat.at(2);
        float z = -quat.at(3);

        float x2 = x * x;
        float y2 = y * y;
        float z2 = z * z;
        float w2 = w * w;
        float xy = x * y;
        float xz = x * z;
        float yz = y * z;
        float wx = w * x;
        float wy = w * y;
        float wz = w * z;

        projected_gravity.at(0) = -2 * (xz + wy);
        projected_gravity.at(1) = -2 * (yz - wx);
        projected_gravity.at(2) = -(w2 - x2 - y2 + z2);
        }
};