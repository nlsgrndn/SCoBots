import argparse
import sys
import gym
from gym import wrappers, logger
from rtpt import RTPT
import matplotlib.pyplot as plt
from gym.envs.classic_control import rendering
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import multiprocessing as mp
import os
import numpy as np

episode_count = 1000
# Create RTPT object
rtpt = RTPT(name_initials='TRo', experiment_name='DataGatherer', max_iterations=episode_count)

# Start the RTPT tracking
rtpt.start()


class GymRenderer():
    def __init__(self, env, record=True, title="video"):
        self.viewer = rendering.SimpleImageViewer()
        self.env = env
        self.record = record
        if record:
            self.video_rec = VideoRecorder(env.env, path=f"videos/{title}.mp4")
            # create videos directory if it does not exist
            if not os.path.exists("videos"):
                os.makedirs("videos")
            print("Videos will be stored in videos directory.")

    def repeat_upsample(self, rgb_array, k=4, l=4, err=[]):
        # repeat kinda crashes if k/l are zero
        if rgb_array is None:
            raise ValueError("The rgb_array is None, probably mushroom_rl bug")
        if k <= 0 or l <= 0:
            if not err:
                print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
                err.append('logged')
            return rgb_array

        # repeat the pixels k times along the y axis and l times along the x axis
        # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

        return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

    def render(self, mode="zoomed"):
        if self.record:
            # self.env.render()
            self.video_rec.capture_frame()
        elif mode == "zoomed":
            rgb = self.env.render('rgb_array')
            upscaled = self.repeat_upsample(rgb, 4, 4)
            self.viewer.imshow(upscaled)
        else:
            self.env.render()

    def close_recorder(self):
        if self.record:
            self.video_rec.close()
            self.video_rec.enabled = False


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def gather_agent(cfg, agent_id):
    args_env_id = 'Pong-v0' #'SpaceInvaders-v0'
    env = gym.make(args_env_id)

    agent = RandomAgent(env.action_space)


    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        episode_id = agent_id * episode_count + i
        renderer = GymRenderer(env, record = True, title=f'{args_env_id}_ep{episode_id:06}')
        step = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            renderer.render()
            # plt.imshow(env.render('rgb_array'))
            # plt.savefig(f"{out_dir}{args_env_id}_ep{agent_id*episode_count+i}_st{step}.png")
            step += 1
            if done:
                break
        renderer.close_recorder()
        if agent_id == 0:
            rtpt.step(subtitle=f"step={i}/{episode_count}")

    env.close()


NUM_PROCESSES = 16


def gather(cfg):
    for agent_id in range(NUM_PROCESSES):
        proc = mp.Process(target=gather_agent, args=(cfg, agent_id))
        proc.start()

    gather_agent(cfg, NUM_PROCESSES)
