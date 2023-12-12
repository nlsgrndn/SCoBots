import torch
import pandas as pd
import numpy as np
from ocatari.ram.pong import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_HUD_PONG
from ocatari.ram.boxing import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_HUD_BOXING
from ocatari.ram.skiing import MAX_NB_OBJECTS_HUD as MAX_NB_OBJECTS_HUD_SKIING

no_label_str = "no_label"

label_list_pacman = [no_label_str, "pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost", "eyes",
                     "white_ghost", "fruit", "save_fruit", "life1", "life2", "score", "corner_block"]

label_list_pong = [no_label_str] + sorted(list(MAX_NB_OBJECTS_HUD_PONG.keys()))
#[no_label_str, "player", 'enemy', 'ball', 'enemy_score', 'player_score']

label_list_carnival = [no_label_str, "owl", 'rabbit', 'shooter', 'refill', 'bonus', "duck",
                       "flying_duck", "score", "pipes", "eating_duck", "bullet"]

label_list_boxing = [no_label_str] + sorted(list(MAX_NB_OBJECTS_HUD_BOXING.keys()))

label_list_tennis = [no_label_str, "player", 'enemy', 'ball', 'ball_shadow', 'net', 'logo',
                     'player_score', 'enemy_score']

label_list_space_invaders = [no_label_str] + [f"{side}_score" for side in ['left', 'right']] + [f"enemy_{idx}"
                                                                                              for idx in
                                                                                              range(6)] \
                            + ["space_ship", "player", "block", "bullet"]

label_list_riverraid = [no_label_str, "player", 'fuel_gauge', 'fuel', 'lives', 'logo', 'score', 'shot', 'fuel_board',
                        'building', 'street', 'enemy']

label_list_air_raid = [no_label_str, "player", 'score', 'building', 'shot', 'enemy']

label_list_skiing = [no_label_str] + sorted(list(MAX_NB_OBJECTS_HUD_SKIING.keys()))


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
        return label_list_pacman
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
