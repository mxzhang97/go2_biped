
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"


#include <cmath>
#include <vector>
#include <array>


#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>


#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>



struct uwb_hri_data
{
    float arm_target_l;
    float arm_target_r;
};


class UWBHRINode: public rclcpp::Node
{
public:
    UWBHRINode();

    void init();
    void run();

private:


    //ros2 
    void init_channels();
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr hri_publisher_;


    //timer
    rclcpp::TimerBase::SharedPtr publish_hri_timer_;



    //socket listeners
    int socket_fd_;
    struct sockaddr_in server_addr_;
    struct sockaddr_in client_addr_;
    void init_socket();


    //callbacks
    void udp_msg_callback(const uwb_hri_data &msg); //runs on udp listener thread 
    void publish_hri_callback(); //runs on executor thread with timer


    void run_process_step(); //runs on rclcpp timed 100hz thread using rclcpp.Rate
    void run_listener_step();


    //threads
    std::thread run_listener_thread_; 
    std::thread run_process_thread_; // Rate-timed


    //locks
    std::mutex ros_msg_lock_; //shared state lock between run_process and timer/executor threads


    //dummy_index
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};

    static const size_t BUFFER_MASK = 127;


    //data
    std::array<uwb_hri_data, 128> hri_data_buffer;

    uwb_hri_data publish_data_;


};


UWBHRINode::UWBHRINode() : Node("uwb_hri_node")
{
    hri_data_buffer.fill({0.0f,0.0f});
}


void UWBHRINode::init(){

    RCLCPP_INFO(this->get_logger(), "Initializing node...");
    init_channels();

    publish_hri_timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&UWBHRINode::publish_hri_callback, this));


    init_socket();


    RCLCPP_INFO(this->get_logger(), "node ok...");
}



void UWBHRINode::init_channels(){
    hri_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/uwb_hri", 10);
}


void UWBHRINode::init_socket(){
    //get the fd for the socket and configure with correct addrs

    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);

    if(socket_fd_<0){
        RCLCPP_FATAL(this->get_logger(), "Failed to get socket...");
        throw std::runtime_error("Failed to get socket in init_socket()...");
    }


    struct timeval read_timeout;
    read_timeout.tv_sec = 0;
    read_timeout.tv_usec = 200000;

    if(setsockopt(socket_fd_, SOL_SOCKET, SO_RCVTIMEO, &read_timeout, sizeof(read_timeout))<0){
        RCLCPP_WARN(this->get_logger(), "Failed to set socket timeout, recvfrom will block on shutdown if waiting for packets...");
    }

    // to clear struct padding 
    std::memset(&server_addr_, 0, sizeof(server_addr_));

    server_addr_.sin_family = AF_INET;
    server_addr_.sin_addr.s_addr = INADDR_ANY;
    server_addr_.sin_port = htons(8080);


    if(bind(socket_fd_, (const struct sockaddr *)&server_addr_, sizeof(server_addr_))<0){
        RCLCPP_FATAL(this->get_logger(), "Failed to bind to port 8080, port in use...");
        close(socket_fd_);
        throw std::runtime_error("Bind failed in init_socket()...");
    }

    RCLCPP_INFO(this->get_logger(), "socket ok on :8080...");

}


void UWBHRINode::udp_msg_callback(const uwb_hri_data &msg){

    //defaults work too, and is strictest (seq_cst)

    //this is ours
    size_t current_head = head_.load(std::memory_order_relaxed);

    size_t next_head = (current_head +1) & BUFFER_MASK;

    hri_data_buffer[next_head] = msg;

    //enforce ordering, store after buffer write
    head_.store(next_head, std::memory_order_release);
    
}



void UWBHRINode::run_listener_step(){


    RCLCPP_INFO(this->get_logger(), "UDP listener started...");

    uwb_hri_data inc_data;

    socklen_t client_len = sizeof(client_addr_);


    while(rclcpp::ok())
    {
        ssize_t len = recvfrom(
            socket_fd_,
            &inc_data,
            sizeof(inc_data),
            0,
            (struct sockaddr *)&client_addr_,
            &client_len
        );

        if(len<0){
            continue;
        }

        if(len == sizeof(uwb_hri_data)){
            udp_msg_callback(inc_data);
        }
        else{
            //cast to int since packets are small
            RCLCPP_WARN(this->get_logger(), "Incoming data size mismatch, expected %d, got %d", (int)sizeof(uwb_hri_data), (int)len);
        }

    }

}


void UWBHRINode::run_process_step(){

    rclcpp::Rate rate(100);


    float filtered_l = 0.0f;
    float filtered_r = 0.0f;


    auto median_filter = [](float a, float b, float c)->float {
        return std::max(std::min(a,b), std::min(std::max(a,b),c));
    };

    while(rclcpp::ok())
    {
        //read head before buffer reads
        size_t current_head = head_.load(std::memory_order_acquire);


        size_t idx_0 = current_head & BUFFER_MASK;
        size_t idx_1 = (current_head - 1) & BUFFER_MASK;
        size_t idx_2 = (current_head - 2) & BUFFER_MASK;

        float raw_l_0, raw_l_1, raw_l_2;
        float raw_r_0, raw_r_1, raw_r_2;

        {


            raw_l_0 = hri_data_buffer[idx_0].arm_target_l;
            raw_l_1 = hri_data_buffer[idx_1].arm_target_l;
            raw_l_2 = hri_data_buffer[idx_2].arm_target_l;

            raw_r_0 = hri_data_buffer[idx_0].arm_target_r;
            raw_r_1 = hri_data_buffer[idx_1].arm_target_r;
            raw_r_2 = hri_data_buffer[idx_2].arm_target_r;

        }

        filtered_l = filtered_l * 0.1 + 0.9 * median_filter(raw_l_0, raw_l_1, raw_l_2);
        filtered_r = filtered_r * 0.1 + 0.9 * median_filter(raw_r_0, raw_r_1, raw_r_2);


        {
            std::lock_guard<std::mutex> lock(ros_msg_lock_);
            publish_data_.arm_target_l = filtered_l;
            publish_data_.arm_target_r = filtered_r;
        }

        rate.sleep();

    }
}


void UWBHRINode::publish_hri_callback(){

    auto msg = std::make_unique<std_msgs::msg::Float32MultiArray>();

    float arm_l, arm_r;

    {
        std::lock_guard<std::mutex> lock(ros_msg_lock_);
        arm_l = publish_data_.arm_target_l;
        arm_r = publish_data_.arm_target_r;
    }

    msg->data = {arm_l, arm_r};

    hri_publisher_->publish(std::move(msg));
}


void UWBHRINode::run()
{
    RCLCPP_INFO(this->get_logger(), "Starting threads...");


    run_listener_thread_ = std::thread(&UWBHRINode::run_listener_step, this);
    run_process_thread_ = std::thread(&UWBHRINode::run_process_step, this);


    RCLCPP_INFO(this->get_logger(), "threads ok...");

    run_listener_thread_.join();
    run_process_thread_.join();
}







int main(int argc, char* argv[]){

    rclcpp::init(argc, argv);


    auto uwb_hri_node = std::make_shared<UWBHRINode>();

    uwb_hri_node->init();
    //init all the channels (we can have init call init_channels)

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(uwb_hri_node);

    std::thread executor_thread([&executor]() {
        executor.spin();
    });


    uwb_hri_node->run();

    executor_thread.join();

    rclcpp::shutdown();

    return 0;
}
