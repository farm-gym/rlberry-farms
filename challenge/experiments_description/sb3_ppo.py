from rlberry.agents import ValueIterationAgent
from rlberry.agents.stable_baselines import StableBaselinesAgent
from stable_baselines3 import PPO

class SBPPOAgent(StableBaselinesAgent):
    def __init__(self, env,**kwargs):
        StableBaselinesAgent.__init__(self, env, algo_cls=PPO, **kwargs)
        
