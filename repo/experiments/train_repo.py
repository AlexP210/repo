from copy import deepcopy

from setup import AttrDict, parse_arguments, set_seed, set_device, setup_logger
from repo.algorithms.repo import Dreamer, MultitaskDreamer, RePo, MultitaskRePo, TIA
from repo.environments import make_env, make_multitask_env


def get_config():
    config = AttrDict()
    config.algo = "repo"
    config.env_id = "dmc-cheetah-jump"
    config.expr_name = "offline-expert-with-eval"
    config.seed = 0
    config.use_gpu = True
    config.gpu_id = 0
    config.out_folder = "/home/alexpl/scratch/repo_training"

    # Integration with TSD
    config.use_tsd_encoder = True
    config.tsd_configuration_path = "/home/alexpl/tsd-project/agents/tsd/tsd/configs/evaluate_teacher.yaml"

    # Dreamer
    config.pixel_obs = False
    config.num_steps = 1_000_000
    config.replay_size = 500000
    config.prefill = 5000
    config.train_every = 500
    config.train_steps = 100
    config.eval_every = 5000
    config.checkpoint_every = 25000
    config.log_every = 500
    config.embedding_size = 1376
    config.hidden_size = 512
    config.belief_size = 445
    config.state_size = 67
    config.dense_activation_function = "elu"
    config.cnn_activation_function = "relu"
    config.batch_size = 50
    config.chunk_size = 50
    config.horizon = 15
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.action_noise = 0.0
    config.action_ent_coef = 3e-4
    config.latent_ent_coef = 0.0
    config.free_nats = 3
    config.model_lr = 3e-4
    config.actor_lr = 8e-5
    config.value_lr = 8e-5
    config.grad_clip_norm = 100.0
    config.load_checkpoint = False
    config.load_offline = False
    config.offline_dir = "/home/alexpl/projects/def-rhinehar/alexpl/TDMPC2-Data/datasets"
    config.offline_truncate_size = 1000000
    config.save_buffer = False
    # Our offline training additions
    config.train_offline = True

    # RePo
    config.target_kl = 3.0
    config.beta_lr = 1e-4
    config.init_beta = 1e-5
    config.prior_train_steps = 5

    # Disagreement model
    config.disag_model = False
    config.ensemble_size = 6
    config.disag_lr = 3e-4
    config.disag_coef = 0.0

    # Inverse dynamics
    config.inv_dynamics = False
    config.inv_dynamics_lr = 3e-4
    config.inv_dynamics_hidden_size = 512

    # Multitask
    config.share_repr = False

    # TIA
    config.tia_obs_coef = 1.0
    config.tia_adv_coef = 1.0
    config.tia_reward_train_steps = 1
    return parse_arguments(config)


if __name__ == "__main__":
    config = get_config()
    set_seed(config.seed)
    set_device(config.use_gpu, config.gpu_id)

    # Logger
    logger = setup_logger(config)

    # Environment
    if "multitask" in config.algo:
        env = make_multitask_env(config.env_id, config.seed, config.pixel_obs)
        eval_env = make_multitask_env(config.env_id, config.seed, config.pixel_obs)
    else:
        env = make_env(config.env_id, config.seed, config.pixel_obs)
        eval_env = make_env(config.env_id, config.seed, config.pixel_obs)

    # Sync video distractors
    if getattr(eval_env.unwrapped, "_img_source", None) is not None:
        eval_env.unwrapped._bg_source = deepcopy(env.unwrapped._bg_source)

    # Agent
    if config.algo == "dreamer":
        algo = Dreamer(config, env, eval_env, logger)
    elif config.algo == "repo":
        algo = RePo(config, env, eval_env, logger)
    elif config.algo == "tia":
        algo = TIA(config, env, eval_env, logger)
    elif config.algo == "dreamer_multitask":
        algo = MultitaskDreamer(config, env, eval_env, logger)
    elif config.algo == "repo_multitask":
        algo = MultitaskRePo(config, env, eval_env, logger)
    else:
        raise NotImplementedError("Unsupported algorithm")
    if not config.train_offline:
        algo.train()
    else:
        algo.train_offline()
