import joblib

import os.path as osp

from model import get_model
from utils import Checkpointer
from solver import get_optimizers
from PIL import Image
from torchvision import transforms

relevant_atariari_labels = {"pong": ["player", "enemy", "ball"], "boxing": ["enemy", "player"]}
#relevant_labels_per_game = {"pong": [1, 2, 3], "boxing": [1, 4]}
relevant_labels_per_game = {"pong": [1,2,4]}
# helper class to clean scene from space scene representation
class SceneCleaner():
    def __init__(self, game):
        # initialize SceneCleaner with the game name and relevant labels for the game
        self.game = game
        self.relevant_labels = relevant_labels_per_game[game]
        # initialize last_known with zeros for each relevant label
        self.last_known = [[0, 0] for _ in self.relevant_labels]

    def clean_scene(self, scene):
            """
            Removes objects from the scene that are below a certain threshold and updates the last known position of relevant labels.

            Args:
                scene (dict): A dictionary containing information about objects in the scene.

            Returns:
                list: A list containing the last known position of relevant labels.
            """
            # initialize empty_keys list to keep track of keys with empty values
            empty_keys = []
            # iterate over each key-value pair in the scene dictionary
            for key, val in scene.items():
                # iterate over each z_where value in the list of values for the current key
                for i, z_where in reversed(list(enumerate(val))):
                    # if the z_where value is below a certain threshold, remove it from the list
                    if z_where[3] < -0.75:
                        scene[key].pop(i)
                # if the filtering step led to an empty list of values for the current key, add the key to the empty_keys list
                if len(val) == 0:
                    empty_keys.append(key)
            # remove keys with empty values from the scene dictionary
            for key in empty_keys:
                scene.pop(key)
            # update last_known with the position of the first instance of each relevant label in the scene dictionary
            for i, el in enumerate(self.relevant_labels):
                if el in scene: #object found
                    self.last_known[i] = scene[el][0][2:]
            # return last_known
            return self.last_known


def load_space(cfg, z_classifier_path=None):   
    # get models
    spacetime_model = get_model(cfg)
    spacetime_model.eval()
    # move to cuda when possible
    use_cuda = 'cuda' in cfg.device
    if use_cuda:
        spacetime_model = spacetime_model.to('cuda:0')
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt,
                                load_time_consistency=cfg.load_time_consistency, add_flow=cfg.add_flow)
    optimizer_fg, optimizer_bg = get_optimizers(cfg, spacetime_model)
    if cfg.resume:
        checkpoint = checkpointer.load_last(cfg.resume_ckpt, spacetime_model, optimizer_fg, optimizer_bg, cfg.device)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step'] + 1

    space = spacetime_model.space
    if z_classifier_path is None:
        z_classifier_path =  'classifiers/pong_z_what_classifier.joblib.pkl'
        #f"classifiers/{cfg.exp_name}_space{cfg.arch_type_desc}_seed{cfg.seed}_z_what_classifier.joblib.pkl"
    print("Loading classifier:" , z_classifier_path)
    z_classifier = joblib.load(z_classifier_path)
    # x is the image on device as a Tensor, z_classifier accepts the latents,
    # only_z_what control if only the z_what latent should be used (see docstring)
    transformation = transforms.ToTensor()

    sc = SceneCleaner(cfg.exp_name)
    # return everything useful
    return space, transformation, sc, z_classifier


# helper function to get scene
def get_scene(cfg, observation, space, z_classifier, sc, transformation, use_cuda=False):
    img = Image.fromarray(observation[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
    #x = torch.moveaxis(torch.tensor(np.array(img)), 2, 0)
    t_img = transformation(img)
    if use_cuda:
        t_img = t_img.cuda()
    scene = space.scene_description(t_img, z_classifier=z_classifier,
                                    only_z_what=True)  # class:[(w, h, x, y)]
    scene_list = sc.clean_scene(scene)
    converted_scene_list = []
    for el in scene_list:
        x, y = convert_spacetime_values(cfg, t_img, *el)
        converted_scene_list.append(x)
        converted_scene_list.append(y) 
    if cfg.exp_name == "pong":
        converted_scene_list = converted_scene_list[4:] + converted_scene_list[2:4] + converted_scene_list[:2] # reorder to player, enemy, ball to match order that was used for RAM data
    #if cfg.exp_name == "boxing":
    #    converted_scene_list = converted_scene_list[2:] + converted_scene_list[:2]
    return scene_list, converted_scene_list


# helper function to convert spacetime normnalized output to coordinates
def convert_spacetime_values(cfg, image_array, x, y):
    if x < 1:
        x = int((x + 1)/2*image_array.shape[1])
        y = int((y + 1)/2*image_array.shape[2])
    if cfg.exp_name == "pong":
        # pong formula
        x, y = int(1.238 * x + 48.1), int(1.5625 * y + 14.125) #TODO: check formula
    elif cfg.exp_name == "boxing":
        # boxing formula
        x, y = int(1.36 * x - (50/3)), int(1.692 * y - 63.54) #TODO: check formula
    #print("Placing point at ", x, y)
    return x, y


# helper function to convert env info into custom list
# only used for atariari
# currently only for boxing
#def convert_to_state(cfg, env_info):
#    labels = env_info["labels"]
#    scene_list = []
#    for label_desc in relevant_atariari_labels[cfg.exp_name]:
#        scene_list.append(labels[label_desc + "_x"])
#        scene_list.append(labels[label_desc + "_y"])
#    return scene_list

def convert_to_stateOCAtari(cfg, env):
    #create dict with go.category as key and (go.y, go.x, go.h, go.w, "S" if go.hud else "M", go.category) as label
    labels = {go.category: (go.y, go.x, go.h, go.w, "S" if go.hud else "M", go.category) for go in sorted(env.objects, key=lambda o: str(o))}
    scene_list = []
    scene_list.append(labels["Player"][1])
    scene_list.append(labels["Player"][0])
    scene_list.append(labels["Enemy"][1])
    scene_list.append(labels["Enemy"][0])
    scene_list.append(labels["Ball"][1])
    scene_list.append(labels["Ball"][0])
    return scene_list