
//unitree_ros2_sdk
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/imu_state.hpp"
#include "unitree_go/msg/wireless_controller.hpp"
#include "unitree_api/msg/request.hpp" 
#include "common/motor_crc.h"

//ros2
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

#include <vector>
#include <cmath>

//threading
#include <chrono>
#include <mutex>
#include <atomic>
#include <thread>

//gamepad
#include "gamepad.hpp"
#include "robot_state_interface.hpp"
#include "biped_controller.hpp"
#include "state_machine.hpp"
#include "common/common_types.hpp"



namespace UNITREE_LEGGED_SDK
{
	constexpr double PosStopF = (2.146E+9f);
	constexpr double VelStopF = (16000.0f);
}



class MainPolicyNode : public rclcpp::Node
{
public:

    MainPolicyNode();

    void run();
    void init(); //this will do the MSC switch after first message is received

private:

    //init
    void init_channels();
    void _send_api_request(rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr publisher, int api_id);


    //channels

    //main feed 
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr low_state_subscription_;
    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr low_cmd_publisher_;
    rclcpp::Subscription<unitree_go::msg::WirelessController>::SharedPtr wireless_controller_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr uwb_hri_subscription_;

    //MSC handoff
    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr sport_request_publisher_;
    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr motion_switcher_publisher_;

    //HRI Node
    //TODO

    //timer
    rclcpp::TimerBase::SharedPtr publish_command_timer_;



    //callbacks
    void low_state_callback(const unitree_go::msg::LowState::SharedPtr msg);
    void wireless_controller_callback(const unitree_go::msg::WirelessController::SharedPtr msg);
    void uwb_hri_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

    //500hz with timer
    void publish_command_callback();

    //TODO
    //void hri_callback();



    //policy_step 50hz
    void run_policy_step();


    //threads for async runs
    std::thread run_policy_thread_;

    
    //load and warmup models/initialization
    void load_models_and_params();
    void init_low_cmd();


    void set_low_cmd();


    //locks

    std::mutex low_state_mutex_;
    std::mutex low_cmd_mutex_;

    std::mutex hri_mutex_;
    std::mutex wireless_controller_mutex_;


    std::atomic<bool> first_state_received_{false};


    //unitree message types

    unitree_go::msg::LowState low_state_;
    unitree_go::msg::LowCmd low_cmd_;
    unitree_go::msg::WirelessController wireless_controller_;


    //arm_targets_from_hri

    alignas(64) hri_arm_targets hri_arm_targets_;




    void update_state_machine(GamepadInterface &gamepad_cpy);




    StateMachine state_machine_;
    BipedController biped_controller_;
    RobotStateInterface robot_state_interface_;
    GamepadInterface gamepad_;

};




MainPolicyNode::MainPolicyNode() : Node("main_policy_node")
{
    RCLCPP_INFO(this->get_logger(), "Constructing Main Policy Node...");

    first_state_received_.store(false);

    load_models_and_params();

    init_channels();


    RCLCPP_INFO(this->get_logger(), "Init low_cmd_...");

    init_low_cmd();


    RCLCPP_INFO(this->get_logger(), "Main Policy Node ok...");

}

void MainPolicyNode::run()
{

    RCLCPP_INFO(this->get_logger(), "Spawning policy thread...");

    run_policy_thread_ = std::thread(&MainPolicyNode::run_policy_step, this);

    RCLCPP_INFO(this->get_logger(), "Policy thread ok...");

    run_policy_thread_.join();

}

void MainPolicyNode::init_channels()
{

    RCLCPP_INFO(this->get_logger(), "Initializing ROS2 channels...");


    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();


    low_state_subscription_ = this->create_subscription<unitree_go::msg::LowState>(
        "/lowstate", qos, std::bind(&MainPolicyNode::low_state_callback, this, std::placeholders::_1));
    wireless_controller_subscription_ = this->create_subscription<unitree_go::msg::WirelessController>(
        "/wirelesscontroller", qos, std::bind(&MainPolicyNode::wireless_controller_callback, this, std::placeholders::_1));
    uwb_hri_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/uwb_hri",qos,std::bind(&MainPolicyNode::uwb_hri_callback, this, std::placeholders::_1));



    low_cmd_publisher_ = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", 10);


    sport_request_publisher_ = this->create_publisher<unitree_api::msg::Request>("/api/sport/request", 10);
    motion_switcher_publisher_ = this->create_publisher<unitree_api::msg::Request>("/api/motion_switcher/request", 10);


    publish_command_timer_ = this->create_wall_timer(std::chrono::milliseconds(2), std::bind(&MainPolicyNode::publish_command_callback, this));



    RCLCPP_INFO(this->get_logger(), "ROS2 channels ok...");

}

void MainPolicyNode::init_low_cmd()
{
    low_cmd_.head[0] = 0xFE;
    low_cmd_.head[1] = 0xEF;
    low_cmd_.level_flag = 0xFF;
    low_cmd_.gpio = 0;

    for(int i=0; i<20; i++)
    {
        low_cmd_.motor_cmd[i].mode = (0x01);   // motor switch to servo (PMSM) mode
        low_cmd_.motor_cmd[i].q = (UNITREE_LEGGED_SDK::PosStopF);
        low_cmd_.motor_cmd[i].kp = (0);
        low_cmd_.motor_cmd[i].dq = (UNITREE_LEGGED_SDK::VelStopF);
        low_cmd_.motor_cmd[i].kd = (0);
        low_cmd_.motor_cmd[i].tau = (0);
    }
}

void MainPolicyNode::load_models_and_params()
{
    RCLCPP_INFO(this->get_logger(), "Initializing models...");
    state_machine_ = StateMachine();
    robot_state_interface_ = RobotStateInterface();
    biped_controller_ = BipedController();
    gamepad_ = GamepadInterface();
    biped_controller_.load_model();
    biped_controller_.reset();

    state_machine_.state = StateMachine::STATES::DAMP;



    RCLCPP_INFO(this->get_logger(), "Models ok...");

}


void MainPolicyNode::low_state_callback(const unitree_go::msg::LowState::SharedPtr msg)
{
    {
        std::lock_guard<std::mutex> lock(low_state_mutex_);
        robot_state_interface_.set(*msg);
    }
    if(!first_state_received_.load())
    {
        first_state_received_.store(true);
    }
}

void MainPolicyNode::uwb_hri_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{

    if(msg->data.size() < 2){
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "uwb_hri data incomplete, reverting to 0.2f,0.2f...");
        hri_arm_targets_.arm_target_l.store(0.2f);
        hri_arm_targets_.arm_target_r.store(0.2f);
        return;
    }


    float target_l = msg->data[0];
    float target_r = msg->data[1];

    if(!std::isfinite(target_l) || !std::isfinite(target_r)){
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "NANs in uwb_hri data, reverting to 0.2f, 0.2f....");
        hri_arm_targets_.arm_target_l.store(0.2f);
        hri_arm_targets_.arm_target_r.store(0.2f);
        return;
    }

    auto clamp_target = [] (float val)->float {
        if(val<0.0f || val>0.6f){
            val = 0.2f;
        }
        return val;
    };


    float clamped_l = clamp_target(target_l);
    float clamped_r = clamp_target(target_r);

    hri_arm_targets_.arm_target_l.store(clamped_l);
    hri_arm_targets_.arm_target_r.store(clamped_r);

}

void MainPolicyNode::publish_command_callback()
{
    {
        std::lock_guard<std::mutex> lock(low_cmd_mutex_);
        low_cmd_publisher_->publish(low_cmd_);
    }
}

void MainPolicyNode::set_low_cmd()
{
    for(int i=0;i<12;i++){
        low_cmd_.motor_cmd[i].q = robot_state_interface_.jpos_des[i];
        low_cmd_.motor_cmd[i].dq = robot_state_interface_.jvel_des[i];
        low_cmd_.motor_cmd[i].kp = robot_state_interface_.kp[i];
        low_cmd_.motor_cmd[i].kd = robot_state_interface_.kd[i];
        low_cmd_.motor_cmd[i].tau = robot_state_interface_.tau_ff[i];
    }

    get_crc(low_cmd_);
}

void MainPolicyNode::wireless_controller_callback(const unitree_go::msg::WirelessController::SharedPtr msg)
{
    {
        std::lock_guard<std::mutex> lock(wireless_controller_mutex_);
        gamepad_.update(*msg);
    }
}



void MainPolicyNode::update_state_machine(GamepadInterface &gamepad_cpy)
{

    if(gamepad_cpy.R1.on_press && gamepad_cpy.L1.pressed)
    {
        if(state_machine_.Crouch())
        {
            std::lock_guard<std::mutex> lock(low_state_mutex_);
            biped_controller_.save_jpos(robot_state_interface_);
        }
        robot_state_interface_.jpos_des = biped_controller_.start_pos;
        robot_state_interface_.jvel_des.fill(0.0);
        robot_state_interface_.kp.fill(biped_controller_.stand_kp);
        robot_state_interface_.kd.fill(biped_controller_.stand_kd);
        robot_state_interface_.tau_ff.fill(0.0);

    }
    if (gamepad_cpy.R2.on_press && gamepad_cpy.L1.pressed)
    {
        if (state_machine_.Stand()) 
        {
            std::lock_guard<std::mutex> lock(low_state_mutex_);
            biped_controller_.save_jpos(robot_state_interface_);
        }
        robot_state_interface_.jpos_des = biped_controller_.start_pos;
        robot_state_interface_.jvel_des.fill(0.0);
        robot_state_interface_.kp.fill(biped_controller_.stand_kp);
        robot_state_interface_.kd.fill(biped_controller_.stand_kd);
        robot_state_interface_.tau_ff.fill(0.0);
    }
    if (gamepad_cpy.A.on_press && gamepad_cpy.L1.pressed)
    {
        if (state_machine_.Policy())
        {
            std::lock_guard<std::mutex> lock(low_state_mutex_);
            biped_controller_.save_jpos(robot_state_interface_);
        }
        robot_state_interface_.jpos_des = biped_controller_.start_pos;
        robot_state_interface_.jvel_des.fill(0.0);
        robot_state_interface_.kp.fill(biped_controller_.ctrl_kp);
        robot_state_interface_.kd.fill(biped_controller_.ctrl_kd);
        robot_state_interface_.tau_ff.fill(0.0);

        biped_controller_.reset();
    }
    if (gamepad_cpy.Y.pressed && gamepad_cpy.L1.pressed)
    {
        state_machine_.Damp();
    }

}


void MainPolicyNode::run_policy_step()
{
    rclcpp::Rate rate(50);

    //profiling
    auto last_loop_time = std::chrono::high_resolution_clock::now();
    double avg_compute_time = 0.0;
    double avg_loop_rate = 0.0;
    int tick_counter =0;

    while(rclcpp::ok())
    {

        auto compute_start_time = std::chrono::high_resolution_clock::now();


        GamepadInterface gamepad_cpy;
        {
            std::lock_guard<std::mutex> lock(wireless_controller_mutex_);
            gamepad_cpy = gamepad_;
            gamepad_.reset();
        }

        update_state_machine(gamepad_cpy);

        {
            std::lock_guard<std::mutex> lock(low_state_mutex_);
            biped_controller_.get_input(robot_state_interface_, gamepad_cpy, hri_arm_targets_);
        }

        if(state_machine_.state == StateMachine::STATES::CROUCH)
        {
            state_machine_.Crouching(biped_controller_);

            robot_state_interface_.jpos_des = biped_controller_.jpos_des;
            robot_state_interface_.jvel_des.fill(0.);
            robot_state_interface_.kp.fill(biped_controller_.stand_kp);
            robot_state_interface_.kd.fill(biped_controller_.stand_kd);
            robot_state_interface_.tau_ff.fill(0.);

        }
        if(state_machine_.state == StateMachine::STATES::STAND)
        {
            state_machine_.Standing(biped_controller_);
            robot_state_interface_.jpos_des = biped_controller_.jpos_des;
            robot_state_interface_.jvel_des.fill(0.);
            robot_state_interface_.kp.fill(biped_controller_.stand_kp);
            robot_state_interface_.kd.fill(biped_controller_.stand_kd);
            robot_state_interface_.tau_ff.fill(0.);

            biped_controller_.warmup();
        }
        if(state_machine_.state == StateMachine::STATES::DAMP)
        {
            robot_state_interface_.jpos_des.fill(0.);
            robot_state_interface_.jvel_des.fill(0.);
            robot_state_interface_.kp.fill(0.);
            robot_state_interface_.kd.fill(5.0);
            robot_state_interface_.tau_ff.fill(0.);
        }

        if(state_machine_.state == StateMachine::STATES::POLICY){

            bool termination;
            //lock to read jpos/proj_grav
            {
                std::lock_guard<std::mutex> lock(low_state_mutex_);
                termination = state_machine_.CheckSafetyBoundaries(robot_state_interface_, this->get_logger());
            }

            if(termination)
            {
                RCLCPP_WARN(this->get_logger(), "Safety bounds exceeded, damping...");
                state_machine_.Damp();
                robot_state_interface_.jpos_des.fill(0.);
                robot_state_interface_.jvel_des.fill(0.);
                robot_state_interface_.kp.fill(0.);
                robot_state_interface_.kd.fill(5.0);
                robot_state_interface_.tau_ff.fill(0.);
            }
            else
            {
                biped_controller_.forward();
                robot_state_interface_.jpos_des = biped_controller_.jpos_des;
                robot_state_interface_.jvel_des.fill(0.);
                robot_state_interface_.kp.fill(biped_controller_.ctrl_kp);
                robot_state_interface_.kd.fill(biped_controller_.ctrl_kd);
                robot_state_interface_.tau_ff.fill(0.);
            }

        }
        //safely write to buffer
        {
            std::lock_guard<std::mutex> lock(low_cmd_mutex_);
            set_low_cmd();
        }

        auto compute_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> compute_ms = compute_end_time - compute_start_time;
        std::chrono::duration<double> loop_s = compute_end_time - last_loop_time;
        last_loop_time = compute_end_time;

        if(tick_counter>0){
            avg_compute_time = avg_compute_time * 0.9 + compute_ms.count() * 0.1;

            double current_hz = (loop_s.count()>0) ? (1.0/loop_s.count()) : 0.0;
            avg_loop_rate = avg_loop_rate * 0.9 + current_hz * 0.1;
        }

        if(tick_counter % 100 == 0) //print every 2s
        {
            std::cout << "Loop Rate: " << avg_loop_rate << "HZ, "
                      << "Compute Time: " << avg_compute_time << "ms, "
                      << "State: " << (int)state_machine_.state << std::endl;
        }

        tick_counter++;





        rate.sleep();

    }


}

void MainPolicyNode::init()
{


    constexpr int API_ID_SPORT_STAND_DOWN = 1005;
    constexpr int API_ID_MOTION_RELEASE_MODE = 1003;


    RCLCPP_INFO(this->get_logger(), "Waiting for first low_state message....");

    while(rclcpp::ok() && !first_state_received_.load()){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if(!rclcpp::ok())
    {
        return;
    }

    RCLCPP_INFO(this->get_logger(), "First low_state received, initializing on bot....");

    RCLCPP_INFO(this->get_logger(), "Sitting down....");

    _send_api_request(sport_request_publisher_, API_ID_SPORT_STAND_DOWN);

    std::this_thread::sleep_for(std::chrono::seconds(2));

    RCLCPP_INFO(this->get_logger(), "Releasing control...");

    _send_api_request(motion_switcher_publisher_, API_ID_MOTION_RELEASE_MODE);

    std::this_thread::sleep_for(std::chrono::seconds(2));

    RCLCPP_INFO(this->get_logger(), "Init ok...");

}

void MainPolicyNode::_send_api_request(rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr publisher, int api_id)
{
    //this is well.... AI generated and overkills with zero-cp unique ptr, but we'll keep it since now we had to learn about unique_ptrs and move
    auto req_msg = std::make_unique<unitree_api::msg::Request>();
    req_msg->header.identity.api_id = api_id;
    req_msg->parameter = "";
    publisher->publish(std::move(req_msg));


    //can also do
    /*
    auto req_msg = unitree_api::msg::Request();
    req_msg.header.identity.api_id = api_id;
    req_msg.parameter = "";
    publisher->publish(req_msg);
    */
}




int main(int argc, char* argv[])
{
    // 1. Initialize the ROS2 system.
    rclcpp::init(argc, argv);

    // 2. Create the main policy node as a shared pointer.
    auto main_policy_node = std::make_shared<MainPolicyNode>();

    // 3. Create a multi-threaded executor to handle ROS2 callbacks.
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(main_policy_node);

    // 4. Spin the executor in a separate thread.
    // This allows ROS callbacks to be processed concurrently in the background.
    std::thread executor_thread([&executor]() {
        executor.spin();
    });


    // put us into MSC manual mode using the req/res apis 
    main_policy_node->init();

    // 5. Run the node's main logic (which spawns and manages its own threads).
    // This call will typically block until the node is ready to shut down.
    main_policy_node->run();

    // 6. Wait for the executor thread to finish.
    executor_thread.join();

    // 7. Shut down the ROS2 system.
    rclcpp::shutdown();

    return 0;
}
