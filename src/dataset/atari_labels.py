import torch
import pandas as pd
import numpy as np
from ocatari.ram.pong import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_PONG
from ocatari.ram.pong import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_PONG
from ocatari.ram.boxing import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_BOXING
from ocatari.ram.boxing import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_BOXING
from ocatari.ram.tennis import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_TENNIS
from ocatari.ram.tennis import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_TENNIS
from ocatari.ram.skiing import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_SKIING
from ocatari.ram.skiing import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_SKIING
from ocatari.ram.carnival import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_CARNIVAL
from ocatari.ram.carnival import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_CARNIVAL
from ocatari.ram.spaceinvaders import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_SPACE_INVADERS
from ocatari.ram.spaceinvaders import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_SPACE_INVADERS
from ocatari.ram.riverraid import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_RIVER_RAID
from ocatari.ram.riverraid import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_RIVER_RAID
from ocatari.ram.mspacman import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_MSPACMAN
from ocatari.ram.mspacman import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_MSPACMAN
from ocatari.ram.carnival import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_ALL_CARNIVAL
from ocatari.ram.carnival import MAX_NB_OBJECTS as MAX_NB_OBJECTS_MOVING_CARNIVAL


no_label_str = "no_label"

label_list_carnival = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_CARNIVAL.keys()))
label_list_carnival_moving = sorted(list(MAX_NB_OBJECTS_MOVING_CARNIVAL.keys()))
moving_indices_carnival = [label_list_carnival.index(moving_label) for moving_label in label_list_carnival_moving]

label_list_mspacman = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_MSPACMAN.keys()))
label_list_mspacman_moving = sorted(list(MAX_NB_OBJECTS_MOVING_MSPACMAN.keys()))
moving_indices_mspacman = [label_list_mspacman.index(moving_label) for moving_label in label_list_mspacman_moving]

label_list_pong = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_PONG.keys()))
label_list_pong_moving = sorted(list(MAX_NB_OBJECTS_MOVING_PONG.keys()))
moving_indices_pong = [label_list_pong.index(moving_label) for moving_label in label_list_pong_moving]

label_list_boxing = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_BOXING.keys()))
label_list_boxing_moving = sorted(list(MAX_NB_OBJECTS_MOVING_BOXING.keys()))
moving_indices_boxing = [label_list_boxing.index(moving_label) for moving_label in label_list_boxing_moving]

label_list_tennis = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_TENNIS.keys()))
label_list_tennis_moving = sorted(list(MAX_NB_OBJECTS_MOVING_TENNIS.keys()))
moving_indices_tennis = [label_list_tennis.index(moving_label) for moving_label in label_list_tennis_moving]                                                   

label_list_space_invaders = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_SPACE_INVADERS.keys()))
label_list_space_invaders_moving = sorted(list(MAX_NB_OBJECTS_MOVING_SPACE_INVADERS.keys()))
moving_indices_space_invaders = [label_list_space_invaders.index(moving_label) for moving_label in label_list_space_invaders_moving]

label_list_riverraid = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_RIVER_RAID.keys()))
label_list_riverraid_moving = sorted(list(MAX_NB_OBJECTS_MOVING_RIVER_RAID.keys()))
moving_indices_riverraid = [label_list_riverraid.index(moving_label) for moving_label in label_list_riverraid_moving]

label_list_skiing = [no_label_str] + sorted(list(MAX_NB_OBJECTS_ALL_SKIING.keys()))
label_list_skiing_moving = sorted(list(MAX_NB_OBJECTS_MOVING_SKIING.keys()))
moving_indices_skiing = [label_list_skiing.index(moving_label) for moving_label in label_list_skiing_moving]


label_list_air_raid = [no_label_str, "player", 'score', 'building', 'shot', 'enemy'] #TODO: Remove or find OCAtari labels


def filter_relevant_boxes(game, boxes_batch, boxes_gt):
    if "MsPacman" in game:
        return [box_bat[box_bat[:, 1] < 104 / 128] for box_bat in boxes_batch]
    elif "Carnival" in game:
        return [box_bat[box_bat[:, 0] > 15 / 128] for box_bat in boxes_batch]
    elif "SpaceInvaders" in game:
        return [box_bat[box_bat[:, 1] > 16 / 128] if len(box_gt[box_gt[:, 1] < 16 / 128]) <= 1 else box_bat for
                box_bat, box_gt in zip(boxes_batch, boxes_gt)]
    elif "Pong" in game:
        # ensure that > 21/128 and < 110/128
        return [box_bat[(box_bat[:, 1] > 21 / 128) & (box_bat[:, 0] > 4 / 128)] for box_bat in boxes_batch]
    elif "Boxing" in game:
        return [box_bat[(box_bat[:, 0] > 19 / 128) * (box_bat[:, 1] < 110 / 128)] for box_bat in boxes_batch]
    elif "Airraid" in game:
        return [box_bat[box_bat[:, 0] > 8 / 128] for box_bat in boxes_batch]
    elif "Riverraid" in game:
        return [box_bat[box_bat[:, 0] < 98 / 128] for box_bat in boxes_batch]  # Does not cover Fuel Gauge
    elif "Tennis" in game:
        return [box_bat[np.logical_or((box_bat[:, 0] > 8 / 128) * (box_bat[:, 1] < 60 / 128),
                                      (box_bat[:, 0] > 68 / 128) * (box_bat[:, 1] < 116 / 128))]
                for box_bat in boxes_batch]
    elif "Skiing" in game:
        return [box_bat for box_bat in boxes_batch] #TODO Find helpful rules if necessary
    else:
        raise ValueError(f"Game {game} could not be found in labels")
    

def get_moving_indices(game):
    if "MsPacman" in game:
        return moving_indices_mspacman
    elif "Carnival" in game:
        return moving_indices_carnival
    elif "SpaceInvaders" in game:
        return moving_indices_space_invaders
    elif "Pong" in game:
        return moving_indices_pong
    elif "Boxing" in game:
        return moving_indices_boxing
    elif "Airraid" in game:
        return ValueError("Moving indices for Airraid not implemented")
    elif "Riverraid" in game:
        return moving_indices_riverraid
    elif "Tennis" in game:
        return moving_indices_tennis
    elif "Skiing" in game:
        return moving_indices_skiing
    else:
        raise ValueError(f"Game {game} could not be found in labels")


def to_relevant(game, labels_moving):
    """
    Return Labels from line in csv file
    """
    no_label_idx = label_list_for(game).index(no_label_str)
    relevant_idx = [[l_m != no_label_idx for l_m in labels_seq] for labels_seq in labels_moving]
    return relevant_idx, [[l_m[rel_idx] for l_m, rel_idx in zip(labels_seq, rel_seq)]
                          for labels_seq, rel_seq in zip(labels_moving, relevant_idx)]


def label_list_for(game):
    """
    Return Labels from line in csv file
    """
    game = game.lower()
    if "mspacman" in game:
        return label_list_mspacman
    elif "carnival" in game:
        return label_list_carnival
    elif "pong" in game:
        return label_list_pong
    elif "boxing" in game:
        return label_list_boxing
    elif "tennis" in game:
        return label_list_tennis
    elif "air" in game and "raid" in game:
        return label_list_air_raid
    elif "riverraid" in game:
        return label_list_riverraid
    elif "space" in game and "invaders" in game:
        return label_list_space_invaders
    elif "skiing" in game:
        return label_list_skiing
    else:
        raise ValueError(f"Game {game} could not be found in labels")


#  Deprecated in favor of IOU
#def labels_for_batch(boxes_batch, entity_list, row, label_list, pieces=None):
#    if pieces is None:
#        pieces = {}
#    bbs = (boxes_batch[:, :4] * (210, 210, 160, 160)).round().astype(int)
#    for en in entity_list:
#        if (f'{en}_visible' not in row or row[f'{en}_visible'].item()) and f'{en}_y' in row and f'{en}_x' in row:
#            pieces[en] = (row[f'{en}_y'].item(), row[f'{en}_x'].item())
#    return labels_for_pieces(bbs, row, pieces, label_list)
#
#
#def labels_for_pieces(bbs, row, pieces, label_list):
#    labels = []
#    for bb in bbs:
#        label = label_for_bb(bb, row, pieces)
#        labels.append(label)
#    return torch.LongTensor([label_list.index(lab) for lab in labels])
#
#
#def label_for_bb(bb, row, pieces):
#    label = min(((name, bb_dist(bb, pos)) for name, pos in pieces.items()), key=lambda tup: tup[1])
#    if label[1] > 15:  # dist
#        label = (no_label_str, 0)
#    label = label[0]  # name
#    if f'{label}_blue' in row and row[f'{label}_blue'].item():
#        label = "blue_ghost"
#    elif f'{label}_white' in row and row[f'{label}_white'].item():
#        label = "white_ghost"
#    return label
#
#
## TODO: Validate mean method with proper labels
#def bb_dist(bb, pos):
#    return abs(bb[0] - pos[0]) + abs(bb[2] - pos[1])
