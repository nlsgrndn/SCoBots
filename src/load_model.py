import joblib
import os.path as osp
from engine.utils import get_config
from model import get_model
from utils import Checkpointer
from solver import get_optimizers
# from utils_rl import Atari
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from vis.utils import fill_image_with_scene
import gym

cfg, task = get_config()


def sprint(scene):
    print("-"*10)
    for obj in scene:
        print([float("{0:.2f}".format(n)) for n in obj])


def show_scene(image, scene):
    filled_image = fill_image_with_scene(image, scene)
    plt.imshow(np.moveaxis(np.array(filled_image.cpu().detach()), (0, 1, 2), (2, 0, 1)))
    plt.show()


def clean_scene(scene):
    empty_keys = []
    for key, val in scene.items():
        for i, z_where in reversed(list(enumerate(val))):
            if z_where[3] < -0.75:
                scene[key].pop(i)
        if len(val) == 0:
            empty_keys.append(key)
    for key in empty_keys:
        scene.pop(key)
    scene_list = []
    for el in [1, 2, 3]:
        if el in scene:
            scene_list.append(scene[el][0][2:])
        else:
            scene_list.append([0, 0]) # object not found
    return scene_list


model = get_model(cfg)
model = model.to('cuda:0')
# state_dicts = torch.load(cfg.resume_ckpt, map_location="cuda:0")
#
# model.space.load_state_dict(state_dicts['model'])

cfg.device_ids = [0]

checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt,
                            load_time_consistency=cfg.load_time_consistency, add_flow=cfg.add_flow)
optimizer_fg, optimizer_bg = get_optimizers(cfg, model)

if cfg.resume:
    checkpoint = checkpointer.load_last(cfg.resume_ckpt, model, optimizer_fg, optimizer_bg, cfg.device)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step'] + 1



space = model.space
z_classifier_path = f"classifiers/{cfg.exp_name}_z_what_classifier.joblib.pkl"
z_classifier = joblib.load(z_classifier_path)
# x is the image on device as a Tensor, z_classifier accepts the latents,
# only_z_what control if only the z_what latent should be used (see docstring)

transformation = transforms.ToTensor()
import matplotlib; matplotlib.use("Tkagg")

env_name = cfg.gamelist[0]
# env = Atari(env_name)
env = gym.make(env_name)
env.reset()
nb_action = env.action_space.n
for i in range(5000):
    observation, reward, done, info = env.step(np.random.randint(nb_action))
    if i % 100 == 0:
        img = Image.fromarray(observation[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
        x = torch.moveaxis(torch.tensor(np.array(img)).cuda(), 2, 0)
        x = transformation(img).cuda()
        scene = space.scene_description(x, z_classifier=z_classifier,
                                        only_z_what=True)  # class:[(w, h, x, y)]
        scene_list = clean_scene(scene)
        sprint(scene_list)
        show_scene(x, scene)
    if done:
        env.reset()