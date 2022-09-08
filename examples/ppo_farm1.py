"""
PPO on Farm1
============
"""

from rlberry.agents.torch import PPOAgent
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
from rlberry_farms import Farm1
from rlberry.agents.torch.utils.training import model_factory_from_env
import numpy as np

policy_configs = {
    "type": "MultiLayerPerceptron",  # A network architecture
    "layer_sizes": (8, 8),  # Network dimensions
    "reshape": False,
    "is_policy": True,
}

value_configs = {
    "type": "MultiLayerPerceptron",
    "layer_sizes": (8, 8),
    "reshape": False,
    "out_size": 1,
}
env_ctor, env_kwargs = Farm1, {"enable_tensorboard": True, "output_dir": "ppo1_results"}


if __name__ == "__main__":
    manager = AgentManager(
        PPOAgent,
        (env_ctor, env_kwargs),
        agent_name="PPOAgent",
        init_kwargs=dict(
            policy_net_fn=model_factory_from_env,
            policy_net_kwargs=policy_configs,
            value_net_fn=model_factory_from_env,
            value_net_kwargs=value_configs,
            learning_rate=1e-3,
            n_steps=128,
            batch_size=14,
            eps_clip=0.2,
        ),
        fit_budget=5e4,
        eval_kwargs=dict(eval_horizon=365),
        n_fit=2,
        parallelization="process",
        mp_context="spawn",
        output_dir="ppo1_results",
    )
    manager.fit()
    evaluation = evaluate_agents([manager], n_simulations=128, plot=False).values
    np.savetxt("ppo_farm1.out", np.array(evaluation), delimiter=",")
    data = plot_writer_data("ppo1_results", "episode_rewards", smooth_weight=0.95)
