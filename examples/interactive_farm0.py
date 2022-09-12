"""
Interactive manual agent on Farm0
=================================

You are the agent.

For windows users, they need to install windows-curses beforehand (installable via `pip install windows-curses`).
"""

from rlberry.agents import AgentWithSimplePolicy
from rlberry.manager import AgentManager, evaluate_agents, plot_writer_data
from rlberry_farms.game0_env import Farm0
from rlberry_farms.utils import farmgymobs_to_obs, get_desc_from_value, get_last_monitor_values
from rlberry.agents.torch.utils.training import model_factory_from_env
import numpy as np

from curses import wrapper
import curses

from rlberry.utils.logging import set_level

set_level("WARNING")

env_ctor, env_kwargs = Farm0, {"monitor": True}


class InteractiveAgent(AgentWithSimplePolicy):
    name = "InteractiveAgent"

    def __init__(self, env, **kwargs):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)
        self.action_str = " "

    def fit(self, stdscr, budget=3e5, **kwargs):
        self.stdscr = stdscr
        self.stdscr.clear()
        curses.curs_set(0)

        observation = self.env.reset()
        farmgym_obs = [" " for i in range(len(observation))]
        self.episode_reward = 0
        self.rewards = []
        for ep in range(int(budget)):
            self.stdscr.clear()

            action = self.policy(observation)
            observation, reward, done, info = self.env.step(action)

            self.episode_reward += reward
            if done:
                self.writer.add_scalar("episode_rewards", self.episode_reward, ep)
                self.rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.env.reset()
        stdscr.addstr(40, 50, "Training done. Press q.")
        while True:
            c = stdscr.getch()
            if c == ord("q"):
                self.stdscr.keypad(0)
                curses.echo()
                curses.nocbreak()
                curses.endwin()

    def policy(self, observation):
        stdscr = self.stdscr
        stdscr.addstr(0, 0, "Tomatoes in Montpellier", curses.A_BOLD + curses.A_UNDERLINE)

        stdscr.addstr(2, 0, "Available actions: 0) Do nothing")
        stdscr.addstr(3, 19, "n) Pour nL of water (for n in {1,...,5})")
        stdscr.addstr(4, 19, "6) Harvest the plant")
        stdscr.addstr(5, 19, "")

        # Rewards
        for j in range(len(self.rewards)):
            stdscr.addstr(
                j, 70, "Reward for episode " + str(j) + " is " + str(self.rewards[j])
            )

        # Observations
        stdscr.addstr(10, 0, "Observations:", curses.A_BOLD + curses.A_UNDERLINE)
        for j in range(len(self.env.observations_txt)):
            stdscr.addstr(12 + j, 0, self.env.observations_txt[j])

            if j not in [4, 7]:
                stdscr.addstr(12 + j, 40, str(observation[j]))
            else:
                if j == 4:
                    stdscr.addstr(
                        12 + j,
                        40,
                        str(get_desc_from_value(observation[j], "rain_amount")),
                    )
                elif j == 7:
                    stdscr.addstr(
                        12 + j,
                        40,
                        str(get_desc_from_value(observation[j], "plant_stage")),
                    )
        stdscr.addstr(7, 0, "Last action: ", curses.A_BOLD + curses.A_UNDERLINE)
        stdscr.addstr(7, len("Last action: "), self.action_str + " " * 20) # empty string to clean the line.

        # unobservable monitored values
        stdscr.addstr(28, 0, "Unobservable farm values:", curses.A_BOLD + curses.A_UNDERLINE)

        df = get_last_monitor_values(self.env.writer)
        for j in range(len(df)):
             stdscr.addstr(30 + j, 0, df.iloc[j]['tag'])
             stdscr.addstr(30 + j, 40, str(df.iloc[j]['value']))
        
        # Actions
        while True:
            c = stdscr.getch()
            if c == 48:
                action = 0
                action_str = "Do nothing"
            elif c == 49:
                action = 1
                action_str = "Pour 1L of water"
            elif c == 50:
                action = 2
                action_str = "Pour 2L of water"
            elif c == 51:
                action = 3
                action_str = "Pour 3L of water"
            elif c == 52:
                action = 4
                action_str = "Pour 4L of water"
            elif c == 53:
                action = 5
                action_str = "Pour 5L of water"
            elif c == 54:
                action = 6
                action_str = "Harvest"
            else:
                action = None
                action_str = "Action not recognized"
            self.action_str = action_str
            if action is not None:
                break
        return action


if __name__ == "__main__":
    manager = AgentManager(
        InteractiveAgent,
        (env_ctor, env_kwargs),
        agent_name="InteractiveAgent",
        fit_budget=3e5,
        eval_kwargs=dict(eval_horizon=365),
        n_fit=1,
        output_dir="interactive_results",
    )
    wrapper(manager.fit)
    evaluation = evaluate_agents([manager], n_simulations=128, show=False).values
    np.savetxt("random_farm0.out", np.array(evaluation), delimiter=",")
    data = plot_writer_data("random_results", "episode_rewards", smooth_weight=0.95)


# This template file gives mean evaluation reward 96.
