from .wheelfoot_flat_config import BipedCfgWF, BipedCfgPPOWF


class BipedCfgNP3OWF(BipedCfgPPOWF):
    class algorithm(BipedCfgPPOWF.algorithm):
        cost_limit = 0.25
        penalty_coef = 1.0
        min_penalty_coef = 0.0
        max_penalty_coef = 100.0
        penalty_lr = 0.05
        use_adaptive_penalty = True
        cost_value_loss_coef = 1.0
        normalize_cost_advantage = False
        cost_dof_pos_limit = 1.0
        cost_dof_vel_limit = 0.2
        cost_torque_limit = 0.2
        cost_collision = 1.0
        cost_termination_contact = 2.0
        cost_fall = 2.0
        cost_power_limit = 0.2
        cost_wheel_vel_limit = 0.1
        soft_dof_vel_limit = BipedCfgWF.rewards.soft_dof_vel_limit
        soft_torque_limit = BipedCfgWF.rewards.soft_torque_limit
        contact_force_threshold = 1.0
        termination_contact_force_threshold = 10.0
        fall_projected_gravity_z = -0.1

    class runner(BipedCfgPPOWF.runner):
        algorithm_class_name = "NP3O"
        experiment_name = "WF_TRON1A_NP3O"
