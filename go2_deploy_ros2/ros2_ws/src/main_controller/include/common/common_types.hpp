#pragma once
#include <cmath>
#include <atomic>

struct hri_arm_targets {
    std::atomic<float> arm_target_l{0.2f};
    std::atomic<float> arm_target_r{0.2f};
};

