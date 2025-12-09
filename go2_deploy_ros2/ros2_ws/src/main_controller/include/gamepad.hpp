#pragma once

#include <cmath>
#include "unitree_go/msg/wireless_controller.hpp"





class Button
{
public:
    void update(bool new_state)
    {
        if(new_state && !pressed)
        {
            on_press=true;

        }
        if(!new_state && pressed)
        {
            on_release=true;
        }
        pressed = new_state;
    }

    void reset()
    {
        on_press = false;
        on_release=false;
    }

    bool on_press = false;
    bool pressed = false;    
    bool on_release = false;
};

// A union to interpret a 16-bit integer as individual button flags.
typedef union
{
    struct
    {
        uint8_t R1 : 1;
        uint8_t L1 : 1;
        uint8_t start : 1;
        uint8_t select : 1;
        uint8_t R2 : 1;
        uint8_t L2 : 1;
        uint8_t F1 : 1;
        uint8_t F2 : 1;
        uint8_t A : 1;
        uint8_t B : 1;
        uint8_t X : 1;
        uint8_t Y : 1;
        uint8_t up : 1;
        uint8_t right : 1;
        uint8_t down : 1;
        uint8_t left : 1;
    } components;
    uint16_t value;
} KeySwitchUnion;

class GamepadInterface
{
public:
    GamepadInterface()
    {
        smooth = 0.03f;
        deadzone=0.01f;
    }

    void update(const unitree_go::msg::WirelessController &msg)
    {
        lx = lx * (1 - smooth) + (std::fabs(msg.lx) < deadzone ? 0.0 : msg.lx) * smooth;
        rx = rx * (1 - smooth) + (std::fabs(msg.rx) < deadzone ? 0.0 : msg.rx) * smooth;
        ry = ry * (1 - smooth) + (std::fabs(msg.ry) < deadzone ? 0.0 : msg.ry) * smooth;
        ly = ly * (1 - smooth) + (std::fabs(msg.ly) < deadzone ? 0.0 : msg.ly) * smooth;

        KeySwitchUnion key_union;
        key_union.value = msg.keys;

        R1.update(key_union.components.R1);
        L1.update(key_union.components.L1);
        start.update(key_union.components.start);
        select.update(key_union.components.select);
        R2.update(key_union.components.R2);
        L2.update(key_union.components.L2);
        F1.update(key_union.components.F1);
        F2.update(key_union.components.F2);
        A.update(key_union.components.A);
        B.update(key_union.components.B);
        X.update(key_union.components.X);
        Y.update(key_union.components.Y);
        up.update(key_union.components.up);
        right.update(key_union.components.right);
        down.update(key_union.components.down);
        left.update(key_union.components.left);

        //std::cout << "raw keys in update: " << std::hex <<msg.keys <<std::dec <<std::endl;

    }
    void reset()
    {
        R1.reset(); L1.reset(); start.reset(); select.reset();
        R2.reset(); L2.reset(); F1.reset(); F2.reset();
        A.reset(); B.reset(); X.reset(); Y.reset();
        up.reset(); down.reset();right.reset(); left.reset();
    }

    Button R1;
    Button L1;
    Button start;
    Button select;
    Button R2;
    Button L2;
    Button F1;
    Button F2;
    Button A;
    Button B;
    Button X;
    Button Y;
    Button up;
    Button right;
    Button down;
    Button left;
    float lx = 0.0f, ly=0.0f, rx=0.0f, ry=0.0f;

private:
    float smooth, deadzone;
};



