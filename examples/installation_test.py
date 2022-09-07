"""
Random agent on Farm0
=====================
"""

from rlberry.agents import AgentWithSimplePolicy
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
from rlberry_farms.game0_env import Farm0
from rlberry.agents.torch.utils.training import model_factory_from_env
import numpy as np

env_ctor, env_kwargs = Farm0, {}


class InstallationTestAgent(AgentWithSimplePolicy):
    name = "InstallationTestAgent"

    def __init__(self, env, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

    def fit(self, budget=10, **kwargs):
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
        return 1


if __name__ == "__main__":
    manager = AgentManager(
        InstallationTestAgent,
        (env_ctor, env_kwargs),
        agent_name="InstallationTestAgent",
        fit_budget=10,
        eval_kwargs=dict(eval_horizon=150),
        n_fit=4,
        parallelization="process",
        mp_context="spawn",
    )
    manager.fit()
    evaluation = evaluate_agents([manager], n_simulations=2, show=False).values

    print("Installation test : Done!")

# This template file gives mean evaluation reward 96.
