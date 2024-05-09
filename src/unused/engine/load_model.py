"""
File to load a model and test it on a game
"""

import joblib
import os.path as osp
import os
from engine.utils import get_config
from  space_models import get_model
from rl_utils import SceneCleaner, load_space
from space_and_moc_utils.checkpointer import Checkpointer
from solver import get_optimizers
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from vis.utils import fill_image_with_scene, place_point
import gymnasium as gym
from pprint import pprint

def sprint(scene):
    print("-"*10)
    for obj in scene:
        print([float("{0:.2f}".format(n)) for n in obj])


def show_scene(image, scene, i, cfg):
    filled_image = fill_image_with_scene(image, scene)
    #plt.imshow(np.moveaxis(np.array(filled_image.cpu().detach()), (0, 1, 2), (2, 0, 1)))
    # also save image
    import os
    if not osp.exists("images"):
        os.makedirs("images")
    if not osp.exists(f"images/{cfg.exp_name}"):
        os.makedirs(f"images/{cfg.exp_name}")
    plt.imsave(f"images/{cfg.exp_name}/{i}.png", np.moveaxis(np.array(filled_image.cpu().detach()), (0, 1, 2), (2, 0, 1)))
    #plt.show()


def load_model(cfg):
    space, transformation, sc, z_classifier = load_space(cfg)
    #import matplotlib; matplotlib.use("Tkagg")
    cfg.device_ids = [0]
    env_name = cfg.gamelist[0]
    env = gym.make(env_name)
    env.reset()
    nb_action = env.action_space.n
    count = 0
    for i in range(1000):
        observation, reward, done, truncated, info = env.step(np.random.randint(nb_action))
        if i % 50 == 0 or count > 0:
            if count >= 10:
                count = 0
                continue
            count += 1
            img = Image.fromarray(observation[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
            #x = torch.moveaxis(torch.tensor(np.array(img)), 2, 0)
            x = transformation(img)
            if 'cuda' in cfg.device:
                x = x.cuda()

            _, log = space.forward(x.unsqueeze(dim=0))
            z_where, z_pres_prob, z_what, z_depth = log['z_where'], log['z_pres_prob'], log['z_what'], log['z_depth']
            z_where, z_what, z_depth = z_where.squeeze().detach().cpu(), z_what.squeeze().detach().cpu(), z_depth.squeeze().detach().cpu()
            z_pres_prob = z_pres_prob.squeeze().detach().cpu()
            z_pres = z_pres_prob > 0.5 # TODO: fix code for case when z_pres_prob <= 0.5 everywhere
            z_pres_prob = z_pres_prob.view(16, 16)
            z_pres = z_pres.view(16, 16)

            z_pres_prob_image = np.array(z_pres)
            z_pres_prob_image = np.repeat(z_pres_prob_image, 8, axis=0)
            z_pres_prob_image = np.repeat(z_pres_prob_image, 8, axis=1)
            z_pres_prob_image = np.expand_dims(z_pres_prob_image, axis=2)
            z_pres_prob_image = np.repeat(z_pres_prob_image, 3, axis=2) 
            z_pres_prob_image[:, :, 1:] = 0
            actual_image = np.moveaxis(np.array(x.cpu().detach()), (0, 1, 2), (2, 0, 1))
            combined_image = 0.5 * actual_image + 0.5 * z_pres_prob_image

            if not osp.exists("images_z_pres_prob"):
                os.makedirs("images_z_pres_prob")
            if not osp.exists(f"images_z_pres_prob/{cfg.exp_name}"):
                os.makedirs(f"images_z_pres_prob/{cfg.exp_name}")
            plt.imsave(f"images_z_pres_prob/{cfg.exp_name}/{i}.png", np.array(combined_image))



            scene = space.scene_description(x, z_classifier=z_classifier,
                                            only_z_what=True)  # class:[(w, h, x, y)]
            scene_list = sc.clean_scene(scene) 
            pprint(scene)
            sprint(scene_list)
            for el in scene_list:
                place_point(x, *el, size=1)
            show_scene(x, scene, i, cfg)
        if done:
            print("Done")
            env.reset()

if __name__ == "__main__":
    cfg, task = get_config()
    load_model(cfg)