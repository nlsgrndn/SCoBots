"""
File to load a model and test it on a game
"""

import joblib
import os.path as osp
from engine.utils import get_config
from model import get_model
from spacetime_rl_algorithms.rl_utils import SceneCleaner, load_space
from utils import Checkpointer
from solver import get_optimizers
# from utils_rl import Atari
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from vis.utils import fill_image_with_scene, place_point
import gym
from pprint import pprint

cfg, task = get_config()


def sprint(scene):
    print("-"*10)
    for obj in scene:
        print([float("{0:.2f}".format(n)) for n in obj])


def show_scene(image, scene):
    filled_image = fill_image_with_scene(image, scene)
    #plt.imshow(np.moveaxis(np.array(filled_image.cpu().detach()), (0, 1, 2), (2, 0, 1)))
    # also save image
    import os
    if not osp.exists("images"):
        os.makedirs("images")
    plt.imsave(f"images/{i}.png", np.moveaxis(np.array(filled_image.cpu().detach()), (0, 1, 2), (2, 0, 1)))
    #plt.show()


use_cuda = 'cuda' in cfg.device
space, transformation, sc, z_classifier = load_space(cfg, z_classifier_path="classifiers/pong_z_what_classifier.joblib.pkl")
#import matplotlib; matplotlib.use("Tkagg")
# env = Atari(env_name)
cfg.device_ids = [0]
env_name = cfg.gamelist[0]
env = gym.make(env_name)
env.reset()
nb_action = env.action_space.n
for i in range(201):
    observation, reward, done, info = env.step(np.random.randint(nb_action))
    if i % 50 == 0:
        img = Image.fromarray(observation[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
        #x = torch.moveaxis(torch.tensor(np.array(img)), 2, 0)
        x = transformation(img)
        if use_cuda:
            x = x.cuda()
        
        _, log = space.forward(x.unsqueeze(dim=0))
        z_where, z_pres_prob, z_what, z_depth = log['z_where'], log['z_pres_prob'], log['z_what'], log['z_depth']
        z_where, z_what, z_depth = z_where.squeeze().detach().cpu(), z_what.squeeze().detach().cpu(), z_depth.squeeze().detach().cpu()
        z_pres_prob = z_pres_prob.squeeze().detach().cpu()
        z_pres = z_pres_prob > 0.5 # TODO: fix code for case when z_pres_prob <= 0.5 everywhere
        z_pres_prob = z_pres_prob.view(16, 16)
        import os
        if not osp.exists("images_z_pres_prob"):
            os.makedirs("images_z_pres_prob")
        plt.imsave(f"images_z_pres_prob/{i}.png", np.array(z_pres_prob))
        plt.imsave(f"images_z_pres_prob/image_{i}.png", np.moveaxis(np.array(x.cpu().detach()), (0, 1, 2), (2, 0, 1)))



        #scene = space.scene_description(x, z_classifier=z_classifier,
        #                                only_z_what=True)  # class:[(w, h, x, y)]
        #scene_list = sc.clean_scene(scene) # remove 
        ## env.render()
        #pprint(scene)
        ##sprint(scene_list)
        #for el in scene_list:
        #    place_point(x, *el, size=1)
        #show_scene(x, scene)
    if done:
        print("Done")
        env.reset()
