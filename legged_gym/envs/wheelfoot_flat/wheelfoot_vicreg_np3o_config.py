from legged_gym.envs.wheelfoot_flat.wheelfoot_np3o_config import (
    BipedCfgWF,
    BipedCfgNP3OWF as _BaseBipedCfgNP3OWF,
)


class BipedCfgNP3OWF(_BaseBipedCfgNP3OWF):
    class MLP_Encoder(_BaseBipedCfgNP3OWF.MLP_Encoder):
        output_detach = True
        num_input_dim = BipedCfgWF.env.num_observations * BipedCfgWF.env.obs_history_length
        explicit_dim = 3       # direct MSE target: scaled base linear velocity
        implicit_dim = 16      # VICReg latent, not directly supervised
        num_output_dim = explicit_dim + implicit_dim
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False

    class algorithm(_BaseBipedCfgNP3OWF.algorithm):
        # Keep the original PPO/NP3O settings and only override encoder loss knobs.
        critic_take_latent = True
        encoder_explicit_dim = 3
        mse_loss_coef = 1.0
        vicreg_loss_coef = 0.05
        vicreg_sim_coef = 25.0
        vicreg_std_coef = 25.0
        vicreg_cov_coef = 1.0
        vicreg_eps = 1.0e-4
