import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import pygame
from .pettingzoo.sisl import pursuit_v4
from gym import spaces

class PursuitEnv(gym.Env):
    def __init__(self, args=None):
        self.args=args
        self._env=pursuit_v4.parallel_env(max_cycles=500, x_size=16, y_size=16,
                                           shared_reward=True, n_evaders=30,
                                           n_pursuers=8, obs_range=args.obs_range, n_catch=2, freeze_evaders=False,
                                           tag_reward=0.01,
                                           catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)
        self.agents=self._env.possible_agents
        self.names=self._env.possible_agents
        # Observations 
        self.observation_space = tuple(self._env.observation_spaces.values())
        self.grid_size=self._env.aec_env.env.env.env.map_matrix.shape
        self.share_observation_space = [spaces.Box(low=np.zeros([3, *self.grid_size]),
                                                  high=np.ones([3, *self.grid_size])*30,
                                                  dtype=np.float64)]*len(self.agents)

        # Actions
        self.action_space = tuple(self._env.action_spaces.values())
        
    
    def seed(self, seed=None):
        self._env.reset(seed)
        
    def _clean_obs(self, states):
        return tuple(states.values())
    
    def _decorate_actions(self, actions):
        return dict(zip(self.names, actions))
    
    def _get_global_state(self):
        map_state=self._env.aec_env.env.env.env.map_matrix
        pursuer_state=self._env.aec_env.env.env.env.pursuer_layer.global_state
        evader_state=self._env.aec_env.env.env.env.evader_layer.global_state
        return np.stack([map_state, pursuer_state, evader_state])

    def reset(self, seed=None):
        states=self._env.reset(seed)
        return self._clean_obs(states), self._get_global_state()
        
    def step(self, actions: tuple):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        actions=actions.squeeze()
        actions=self._decorate_actions(actions)
        obs, reward, done, info = self._env.step(actions)
        global_state=self._get_global_state()
        return tuple(obs.values()), global_state, tuple(reward.values()), tuple(done.values()), tuple(info.values())

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()