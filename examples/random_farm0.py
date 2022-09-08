"""
Random agent on Farm0
=====================
"""

from rlberry.agents import AgentWithSimplePolicy
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
from rlberry_farms import Farm0
from rlberry.agents.torch.utils.training import model_factory_from_env
import numpy as np

env_ctor, env_kwargs = Farm0, {}


class RandomAgent(AgentWithSimplePolicy):
    name = "RandomAgent"

    def __init__(self, env, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

    def fit(self, budget=100, **kwargs):
        observation = self.env.reset()
        episode_reward = 0
        for ep in range(int(budget)):
            action = self.policy(observation)
            observation, reward, done, _ = self.env.step(action)
            episode_reward += reward
            if done:
                self.writer.add_scalar("episode_rewards", episode_reward, ep)
                episode_reward = 0
                self.env.reset()

    def policy(self, observation):
        return self.env.action_space.sample()  # choose an action at random


if __name__ == "__main__":
    manager = AgentManager(
        RandomAgent,
        (env_ctor, env_kwargs),
        agent_name="RandomAgent",
        fit_budget=1e4,
        eval_kwargs=dict(eval_horizon=365),
        n_fit=4,
        parallelization="process",
        mp_context="spawn",
        output_dir="random_results",
    )
    manager.fit()
    evaluation = evaluate_agents([manager], n_simulations=128, plot=False).values
    np.savetxt("random_farm0.out", np.array(evaluation), delimiter=",")
    data = plot_writer_data("random_results", "episode_rewards", smooth_weight=0.95)


# This template file gives mean evaluation reward 96.
