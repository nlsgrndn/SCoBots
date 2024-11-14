from PIL import Image
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import random
from dataset import bb
from motion import flow
from motion import mode
from motion.motion_processing import ProcessingVisualization, BoundingBoxes, \
    ClosingMeanThreshold, IteratedCentroidSelection, Skeletonize, Identity, FlowBoundingBox, ZWhereZPres, \
    set_color_hist, set_special_color_weight
from space_and_moc_utils.niceprint import pprint as print


#OCAtari
from ocatari.core import OCAtari
import random
"""
If you look at the atari_env source code, essentially:

v0 vs v4: v0 has repeat_action_probability of 0.25 (meaning 25% of the time the
previous action will be used instead of the new action),
while v4 has 0 (always follow your issued action)
Deterministic: a fixed frameskip of 4, while for the env without Deterministic,
frameskip is sampled from (2,5)
There is also NoFrameskip-v4 with no frame skip and no action repeat
stochasticity.
"""


def some_steps(agent):
    agent.env.reset()
    for _ in range(10):
        _, _, _, _, _ = take_action(agent)
    return take_action(agent)


def draw_images(obs, image_n, rgb_folder, bgr_folder):
    ## RAW IMAGE
    img = Image.fromarray(obs, 'RGB')
    img.save(f'{rgb_folder}/{image_n:05}.png')
    ## BGR SPACE IMAGES
    img = Image.fromarray(
        obs[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
    img.save(f'{bgr_folder}/{image_n:05}.png')  # better quality than jpg


def take_action(agent):
    action = agent.draw_action()
    obs, reward, done, truncated, info = agent.env.step(action)
    return obs, reward, done, truncated, info

def compute_root_images(imgs, data_base_folder, game):
    img_arr = np.stack(imgs)
    mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=img_arr).astype(np.uint8)
    os.makedirs(f"{data_base_folder}/{game}-v0/background", exist_ok=True)
    frame = Image.fromarray(mode)
    frame.save(f"{data_base_folder}/{game}-v0/background/mode.png")
    print("blue", f"Saved mode.png in {data_base_folder}/{game}-v0/background/")


def create_dataset_using_ocatari():
    parser = argparse.ArgumentParser(
        description='Create the dataset for a specific game of ATARI Gym')
    parser.add_argument('-g', '--game', type=str, help='An atari game',
                        default='SpaceInvaders')
    parser.add_argument('--compute_root_images', default=False, action="store_true",
                        help='instead compute the mode of the images')
    # parser.add_argument('--root', default=True, action="store_true",
    #                     help='use the root-mode (or root-median --median) instead of the trail')
    parser.add_argument('--no_color_hist', default=False, action="store_true",
                        help='use the color_hist to filter')
    parser.add_argument('--render', default=False, action="store_true",
                        help='renders the environment')
    parser.add_argument('-s', '--stacks', default=True, action="store_false",
                        help='should render in correlated stacks of T (defined in code) frames')
    parser.add_argument('--bb', default=True, action="store_false",
                        help='should compute bounding_boxes')
    parser.add_argument('--flow', action="store_true",
                        help='should compute flow information (default False)')
    parser.add_argument('-f', '--folder', type=str, choices=["train", "test", "validation"],
                        required=True,
                        help='folder to write to: train, test or validation')
    parser.add_argument('--vis', default=False, action="store_true",
                        help='creates folder vis with visualizations which can be used for debugging')
    args = parser.parse_args()

    # initialize
    print("box", "Settings:", args)
    folder_sizes = {"train": 2048, "test": 128, "validation": 128}
    limit = folder_sizes[args.folder]
    data_base_folder = "../aiml_atari_data"
    mode_base_folder = "../aiml_atari_data"
    REQ_CONSECUTIVE_IMAGE = 20 # defines how many steps are taken before the last T frames are saved
    T = 4 # number of consecutive frames, ATTENTION: most of the code is written for T=4, so it might not work for other values
    assert T <= REQ_CONSECUTIVE_IMAGE, "T must be smaller or equal to REQ_CONSECUTIVE_IMAGE"

    rgb_folder, bgr_folder, flow_folder, bb_folder, vis_folder, mode_folder = \
        create_folders(args, data_base_folder)

    # for optional visualization
    visualizations_flow = [
        Identity(vis_folder, "Flow", max_vis=50, every_n=1),
    ] if args.vis else []
    visualizations_mode = [
        Identity(vis_folder, "Mode", max_vis=50, every_n=1),
        ZWhereZPres(vis_folder, "Mode", max_vis=20, every_n=2),
    ] if args.vis else []
    visualizations_bb = [
        BoundingBoxes(vis_folder, 'BoundingBox', max_vis=20, every_n=1) # '' instead of 'BoundingBox' before
    ] if args.vis else []


    agent, observation, info = configure(args)

    # take some steps to not start at the beginning of the game (might be unnecessary)
    for _ in range(100):
        obs, reward, done, truncated, info = take_action(agent)
        #print(agent.env.objects)
        if done or truncated:
            agent.env.reset()

    # compute the root image i.e. mode as the background
    if args.compute_root_images:
        root_image_limit = 100 # was 1000 before
        imgs = []
        pbar = tqdm(total=root_image_limit)
        while len(imgs) < root_image_limit:
            obs, reward, done, truncated, info = take_action(agent)
            if np.random.rand() < 0.01:
                imgs.append(obs)
                pbar.update(1)
            if done or truncated:
                env.reset()
                for _ in range(100):
                    obs, reward, done, truncated, info = take_action(agent)
        compute_root_images(imgs, data_base_folder, args.game)
        exit(0)

    image_count = 0
    consecutive_images = []
    consecutive_images_info = []


    print("Init steps...")
    for _ in range(50):
        obs, reward, done, truncated, info = take_action(agent)

    mode_path = f"{mode_base_folder}/{args.game}-v0/background"
    if not os.path.exists(f"{mode_path}/mode.png"):
        print("red", f"Couldn't find {mode_path}/mode.png, use --trail to use the trail instead")
        exit(1)
    
    root_mode = np.array(Image.open(f"{mode_base_folder}/{args.game}-v0/background/mode.png"))[:, :, :3]

    if not args.no_color_hist:
        set_color_hist(root_mode)
        # Exceptions where mode-delta is not working well, but it is easily fixed,
        # by marking some colors interesting or uninteresting respectively.
        # Those would be no issue with FlowNet
        if "Pong" in args.game:
            set_special_color_weight(15406316, 8)
        if "Airraid" in args.game:
            set_special_color_weight(0, 20000)
        if "Riverraid" in args.game:
            set_special_color_weight(3497752, 20000)

    pbar = tqdm(total=limit)

    while True:
        obs, reward, done, truncated, info = take_action(agent)

        game_objects = [(go.y, go.x, go.h, go.w, "S" if go.hud else "M", go.category) for go in sorted(agent.env.objects, key=lambda o: str(o))]
        info["bbs"] = game_objects

        if (obs==0).all(): # black screen
            continue
        if args.render:
            env.render()
            
        consecutive_images += [obs]
        consecutive_images_info.append(info)
        if len(consecutive_images) == REQ_CONSECUTIVE_IMAGE:
            space_stack = []
            for frame in consecutive_images[:-T]:
                space_stack.append(frame)
            resize_stack = []
            for i, (frame, img_info) in enumerate(zip(consecutive_images[-T:], consecutive_images_info[-T:])):
                space_stack.append(frame)
                frame_space = Image.fromarray(frame[:, :, ::-1], 'RGB').resize((128, 128), Image.LANCZOS)
                resize_stack.append(np.array(frame_space))
                frame_space.save(f'{bgr_folder}/{image_count:05}_{i}.png')
                img = Image.fromarray(frame, 'RGB')
                img.save(f'{rgb_folder}/{image_count:05}_{i}.png')
                bb.save(args, frame_space, img_info, f'{bb_folder}/{image_count:05}_{i}.csv',
                        visualizations_bb)
            # for i, fr in enumerate(space_stack):
            #     Image.fromarray(fr, 'RGB').save(f'{vis_folder}/Mode/Stack_{image_count:05}_{i:03}.png')
            resize_stack = np.stack(resize_stack)
            space_stack = np.stack(space_stack)

            # save the flow
            if args.flow:
                flow.save(space_stack, f'{flow_folder}/{image_count:05}_{{}}.pt', visualizations_flow)
            
            # save the mode
            mode.save(space_stack, f'{mode_folder}/{image_count:05}_{{}}.pt', visualizations_mode,
                          mode=root_mode, space_frame=resize_stack)
            
            while done or truncated:
                obs, reward, done, truncated, info = some_steps(agent)
                step = 0
            consecutive_images, consecutive_images_info = [], []
            pbar.update(1)
            image_count += 1
        else:
            while done or truncated:
                obs, reward, done, truncated, info = some_steps(agent)
                consecutive_images, consecutive_images_info = [], []
        if image_count == limit:
            break

    print(f"Dataset Generation is completed. Everything is saved in {data_base_folder}.")

def make_deterministic(seed, mdp, states_dict=None):
    if states_dict is None:
        np.random.seed(seed)
        #mdp.env.env.np_random.set_state(seed)
        #mdp.env.env.seed_game(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Set all environment deterministic to seed {seed}")
    else:
        np.random.set_state(states_dict["numpy"])
        torch.random.set_rng_state(states_dict["torch"])
        mdp.env.env.np_random.set_state(states_dict["env"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Reset environment to recovered random state ")

#from hackatari.riverraid import ConstantBackgroundRiverraid
def configure(args):
    global env
    print(f"Playing {args.game}...")
    env = OCAtari(args.game, mode = "revised", hud=True, render_mode="rgb_array") # revised(=ram) mode should be used # ConstantBackgroundRiverraid(env_name="RiverraidDeterministic-v0", mode="revised", hud=True, obs_mode="dqn") 
    observation, info = env.reset()
    make_deterministic(0 if args.folder == "train" else 1 if args.folder == "validation" else 2, env)
    agent = RandomAgent(env)
    return agent, observation, info

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def draw_action(self):
        return random.randint(0, self.env.nb_actions-1)


def create_folders(args, data_base_folder):
    rgb_folder = f"{data_base_folder}/{args.game}-v0/rgb/{args.folder}"
    bgr_folder = f"{data_base_folder}/{args.game}-v0/space_like/{args.folder}" 
    bb_folder = f"{data_base_folder}/{args.game}-v0/bb/{args.folder}" # gt information based on extraction from internals of the game. is used for evaluation
    flow_folder = f"{data_base_folder}/{args.game}-v0/flow/{args.folder}"
    mode_folder = f"{data_base_folder}/{args.game}-v0/mode/{args.folder}"
    if args.vis:
        vis_folder = f"{data_base_folder}/{args.game}-v0/vis/{args.folder}"
    else:
        vis_folder = None
    os.makedirs(bgr_folder, exist_ok=True)
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(flow_folder, exist_ok=True)
    os.makedirs(mode_folder, exist_ok=True)
    os.makedirs(bb_folder, exist_ok=True)
    if args.vis:
        os.makedirs(vis_folder + "/BoundingBox", exist_ok=True)
        os.makedirs(vis_folder + "/Flow", exist_ok=True)
        os.makedirs(vis_folder + "/Mode", exist_ok=True)
    return rgb_folder, bgr_folder, flow_folder, bb_folder, vis_folder, mode_folder


if __name__ == '__main__':
    create_dataset_using_ocatari()
