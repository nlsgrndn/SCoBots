import argparse
import sys
import gym
from gym import wrappers, logger
from rtpt import RTPT
import matplotlib.pyplot as plt
from spacetime_dqn import SpacetimeDQNAgent, model_name
import torch
from ocatari.core import OCAtari
import rl_utils
from gymnasium.wrappers.record_video import RecordVideo
import multiprocessing as mp
import os
import numpy as np

episode_count = 1
# Create RTPT object
rtpt = RTPT(name_initials='TRo', experiment_name='DataGatherer', max_iterations=episode_count)

# Start the RTPT tracking
rtpt.start()

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return self.action_space.sample()



def gather_agent(cfg, agent_id):
    # load environment
    args_env_id = 'Pong-v4'
    env = OCAtari(args_env_id, mode="revised", hud=True, render_mode="rgb_array")

    #load agent
    model_path = model_name(cfg.exp_name)
    agent = SpacetimeDQNAgent(env.action_space.n, torch.device('cpu'),  model_path,)
    space, transformation, sc, z_classifier = rl_utils.load_space(cfg)
    use_cuda = "cuda" in cfg.device

    for i in range(episode_count):

        # reset environment
        env.reset()
        s_state = rl_utils.convert_to_stateOCAtari(cfg, env)
        s_state = torch.tensor(s_state, dtype=torch.float) 
        # state stacking to have current and previous state at once
        state = torch.cat((s_state, s_state), 0)
        env = RecordVideo(env, video_folder=f"videos")
        env.start_video_recorder()
        step = 0

        while True:
            state = state.to(agent.device)
            action = agent.act(state)
            observation, reward, done, truncated, info = env.step(action.item())

            # get state representation for next step
            USE_GT_AS_STATE = False
            if USE_GT_AS_STATE:
                s_next_state = rl_utils.convert_to_stateOCAtari(cfg, env)
            else:
                _, s_next_state = rl_utils.get_scene(cfg, observation, space, z_classifier, sc, transformation, use_cuda)
            s_next_state = torch.tensor(s_next_state, dtype=torch.float)
            state = torch.cat((s_state, s_next_state), 0)
            s_state = s_next_state # store for next step such that s_next_state can be overwritten

            env.render()
            step += 1
            print(step)
            if done or step > 1000:
                break
        env.close_video_recorder()
        print(f"Episode {i} finished after {step} steps.")
        if agent_id == 0:
            rtpt.step(subtitle=f"step={i}/{episode_count}")

    env.close()


NUM_PROCESSES = 1


def gather(cfg):
    #for agent_id in range(NUM_PROCESSES):
    #    proc = mp.Process(target=gather_agent, args=(cfg, agent_id))
    #    proc.start()
    print("ATTENTION: Currently only one episode can be gathered at a time.")
    gather_agent(cfg, NUM_PROCESSES)
