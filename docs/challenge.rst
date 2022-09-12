
.. _challenge:

Scool Challenge
===============

The challenge is on the flanders server.

Once connected to the server, everything is done through the command line utility :code:`farmscool`. Most informations about :code:`farmscool` can be found using :code:`farmscool --help`.

To submit an agent, you have to give as input a python file containing the agent. The agent must be an rlberry agent and it must be called "Agent" (with capital and everything). 

Example file: :code:`agent.py`

.. code:: python

    from rlberry.agents import AgentWithSimplePolicy

    class Agent(AgentWithSimplePolicy):
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

To submit to the leaderboard, use

.. code:: bash
   
   farmscool add agent.py 1000


The number 1000 corresponds to the number of steps (the budget in fit).

- Use :code:`farmscool info` to see how many jobs pending.
- Use :code:`farmscool log` to see the log of the current training.
- Use :code:`farmscool leaderboard` to see the leaderboard.

