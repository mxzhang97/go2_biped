from legged_gym.envs.base.base_config import BaseConfig


class Go2BipedRMStudentQuadCfg(BaseConfig):
    class env: 
        num_envs = 4096 
        num_actions = 12
        num_observations = 54
        num_privileged_observations = 2
        send_timeouts = True
        episode_length_s = 20
        env_spacing = 2.0
        fail_to_terminal_time_s = 0.5

    class terrain:
        mesh_type = 'plane'
        plane_length = 200.0
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 5
        static_friction = 0.8 
        dynamic_friction = 0.6
        restitution = 0.

    class commands:
        num_commands = 5
        mode_resampling = 7
        resampling_time = 10.
        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-1.0, 1.2]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            fl_arm_track_target = [0.1,0.6]
            fr_arm_track_target = [0.1,0.6]

    class init_state:
        pos = [0.0, 0.0, 0.34]  # x,y,z [m]
        rot = [0.0,0.0,0.0,1.0]
        lin_vel = [0.0,0.0,0.0]
        ang_vel = [0.0,0.0,0.0]
        base_ang_random_scale = 0.

        default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.0,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.0,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]
        }

    class control:
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {"joint": 25.0}  # [N*m/rad]
        damping = {"joint": 0.5}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4
        dt = 0.02
        


    class asset:
        # Common: 
        name = "go2"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        obtain_link_contact_states = True
        contact_state_link_names = ["thigh", "calf", "foot"]
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base", "hip", "Head"]
        terminate_after_contacts_on = ["base", "hip", "Head"]
        flip_visual_attachments = False
        self_collisions_gym = 0
        disable_gravity=False
        fix_base_link = False    # fix base link to the world
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        default_dof_drive_mode = 3   # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:

        randomize_friction = True 
        friction_range = [0.2, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 12
        max_push_vel_xy = 1.0
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.10]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = True
        joint_armature_range = [0.015, 0.025] 
        randomize_joint_stiffness = True
        joint_stiffness_range = [0.01, 0.02]
        randomize_joint_damping = True
        joint_damping_range = [0.25, 0.3]
        randomize_ctrl_delay = False
        ctrl_delay_step_range = [0, 1]
        randomize_joint_friction = False
        joint_friction_range = [0.0, 0.1]


    class noise:
        add_noise = True
        noise_level = 1.0 
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            com_cop = 0.05

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
        clip_observations = 100.
        clip_actions = 100.


    class rewards:
        base_height_target = 0.55
        max_contact_force = 500.0
        only_positive_rewards = True
        stage_2 = True
        min_foot_height = 0.06
        min_gait_freq_step = 10
        gait_transition_bonus = 0.6 
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_projected_gravity = -0.1

        class scales:
            tracking_lin_vel = 1.0 
            tracking_ang_vel = 0.6 
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = 0.8
            orientation_3 = -0.03
            torques = -0.
            dof_acc = -0.
            base_height = -0.5
            feet_air_time = 1.0 
            collision = -1.0
            action_rate = -0.02


            inv_pendulum = -0.1
            inv_pendulum_acc = -0.0001
            cart_table_len_xy= -0.1

            fr_contact_force = -0.03
            fl_contact_force = -0.03
            f_contact_force = -0.03


            fr_arm_track = 0.3
            fl_arm_track = 0.3
            fr_arm_track_lateral_distance = -0.4
            fl_arm_track_lateral_distance = -0.4



            rear_hip_joint = -0.2
            front_hip_joint = -0.1



            ##new rewards to get more smoothness (second stage)
            dof_pos_limits = -0.0 #10.0
            action_smoothness = -0.0 #0.01
            dof_vel = -0.0
            dof_acc = -0.0



            #quad/biped transition
            orientation_quad = -2.0
            ang_vel_quad = -0.1
            lin_vel_quad = -0.2
            torque_quad = -2e-4
            joint_vel_quad = -5e-4
            joint_acc_quad = -2e-7
            default_pos_quad = 1.0

    class viewer:
        ref_env = 0
        pos = [2, 2, 2]       # [m]
        lookat = [0., 0, 1.]  # [m]
        rendered_envs_idx = [i for i in range(5)]  # number of environments to be rendered


    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        use_gpu_pipeline = True

        class physx:
            use_gpu = True
            num_subscenes = 0
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
    