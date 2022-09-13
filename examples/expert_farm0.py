"""
Expert agent on Farm0
=====================
"""

from rlberry.agents import AgentWithSimplePolicy
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
from rlberry_farms import Farm0
import numpy as np

env_ctor, env_kwargs = Farm0, {}


class Agent(AgentWithSimplePolicy):
    name = "ExpertAgentFarm0"
    fruit_stage_duration_count = 0

    def __init__(self, env, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

    def fit(self, budget=100, **kwargs):
        observation = self.env.reset()
        episode_reward = 0
        for ep in range(int(budget)):
            action = self.policy(observation)
            observation, reward, done, info = self.env.step(action)
            episode_reward += reward
            if done:
                self.writer.add_scalar("episode_rewards", episode_reward, ep)
                episode_reward = 0
                self.env.reset()

    def policy(self, observation):
        # Policy : When the plant stage go to 'fruit', wait 8 days, and harvest on the 9

        # The actions are :
        #     0) Do nothing
        #     1) Pour 1L of water
        #     2) Pour 5L of water
        #     3) Harvest the plant

        # states for 'stage for plants' (observation[7]) are :
        # 0:'none'
        # 1:'seed'
        # 2:'entered_grow'
        # 3:'grow'
        # 4:'entered_bloom'
        # 5:'bloom'
        # 6:'entered_fruit'
        # 7:'fruit'
        # 8:'entered_ripe'
        # 9:'ripe'
        # 10:'entered_seed'
        # 11:'harvested'
        # 12:'dead'

        # print(self.env.farm.get_free_observations())
        if observation[7] in [6, 7, 8, 9]:
            if self.fruit_stage_duration_count > 8:
                next_action = 3  # harvesting
                self.fruit_stage_duration_count = 0
            else:
                next_action = 1  # watering slowly
                self.fruit_stage_duration_count += 1
        else:
            next_action = 1  # watering slowly

        return next_action


if __name__ == "__main__":
    manager = AgentManager(
        Agent,
        (env_ctor, env_kwargs),
        agent_name="ExpertAgentFarm0",
        fit_budget=1e4,
        eval_kwargs=dict(eval_horizon=365),
        n_fit=4,
        parallelization="process",
        mp_context="spawn",
        output_dir="expert_farm0_results",
    )
    manager.fit()
    evaluation = evaluate_agents([manager], n_simulations=128, plot=False).values
    np.savetxt("expert_farm0.out", np.array(evaluation), delimiter=",")
    data = plot_writer_data(
        "expert_farm0_results", "episode_rewards", smooth_weight=0.95
    )
