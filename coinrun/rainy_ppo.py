import os
from rainy import Config
from rainy.agent import PpoAgent
import rainy.util.cli as cli
from torch.optim import Adam
from rainy.net import ActorCriticNet, DqnConv, LinearHead
from rainy.net.init import Initializer, orthogonal
from rainy.util import Device
from typing import Tuple


def a2c_conv(state_dim: Tuple[int, int, int], action_dim: int, device: Device) -> ActorCriticNet:
    body = DqnConv(
        state_dim,
        kernel_and_strides=[(8, 1), (4, 1), (3, 1)],
        hidden_channels=(32, 64, 32),
        output_dim=256
    )
    ac_head = LinearHead(body.output_dim, action_dim, Initializer(weight_init=orthogonal(0.01)))
    cr_head = LinearHead(body.output_dim, 1)
    return ActorCriticNet(body, ac_head, cr_head, device=device)


def config() -> Config:
    c = Config()
    c.nworkers = 8
    c.set_parallel_env(lambda _env_gen, _num_w: ParallelRogueEnvExt(StairRewardParallel(
        [CONFIG] * c.nworkers,
        max_steps=500,
        stair_reward=50.0,
        image_setting=EXPAND,
    )))
    c.eval_env = RogueEnvExt(StairRewardEnv(
        config_dict=CONFIG,
        max_steps=500,
        stair_reward=50.0,
        image_setting=EXPAND
    ))
    c.set_optimizer(lambda params: Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.set_net_fn('actor-critic', a2c_conv)
    c.max_steps = int(2e7)
    c.grad_clip = 0.5
    c.episode_log_freq = 100
    c.eval_freq = None
    c.eval_deterministic = False
    # ppo parameters
    c.nsteps = 100
    c.value_loss_weight = 0.5
    c.gae_tau = 0.95
    c.use_gae = True
    c.ppo_minibatch_size = 200
    c.ppo_clip = 0.1
    c.lr_decay = True
    return c


if __name__ == '__main__':
    cli.run_cli(config(), PpoAgent, script_path=os.path.realpath(__file__))
