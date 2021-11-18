import torch
import pandas as pd
import numpy as np

label_list_pacman = ["pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost", "eyes",
                     "white_ghost", "fruit", "save_fruit", "life1", "life2", "score", "corner_block", "no_label"]

label_list_pong = ["player", 'enemy', 'ball', 'enemy_score', 'player_score', "no_label"]

label_list_carnival = ["owl", 'rabbit', 'shooter', 'refill', 'bonus', "duck",
                       "flying_duck", "score0", "no_label"]

# Maybe enemy bullets, but how should SPACE differentiate
label_list_space_invaders = [f"{side}_score" for side in ['left', 'right']] + [f"enemy_{idx}"
                                                                               for idx in
                                                                               range(6)] \
                            + ["space_ship", "player", "block", "bullet", "no_label"]


def filter_relevant_boxes(game, boxes_batch, boxes_gt):
    if "MsPacman" in game:
        return [box_bat[box_bat[:, 1] < 103 / 128] for box_bat in boxes_batch]
    elif "Carnival" in game:
        return
    elif "SpaceInvaders" in game:
        return [box_bat[box_bat[:, 1] > 19 / 128] if len(box_gt[box_gt[:, 1] < 19 / 128]) <= 1 else box_bat for
                box_bat, box_gt in zip(boxes_batch, boxes_gt)]
    elif "Pong" in game:
        return [box_bat[box_bat[:, 1] > 19 / 128] for box_bat in boxes_batch]
    else:
        raise ValueError(f"Game {game} could not be found in labels")


def to_relevant(game, labels_moving):
    """
    Return Labels from line in csv file
    """
    if "MsPacman" in game:
        return labels_moving != label_list_pacman.index("no_label")
    elif "Carnival" in game:
        return labels_moving != label_list_carnival.index("no_label")
    elif "Pong" in game:
        return labels_moving != label_list_pong.index("no_label")
    elif "SpaceInvaders" in game:
        return labels_moving != label_list_space_invaders.index("no_label")
    else:
        raise ValueError(f"Game {game} could not be found in labels")


def label_list_for(game):
    """
    Return Labels from line in csv file
    """
    if "MsPacman" in game:
        return label_list_pacman
    elif "Carnival" in game:
        return label_list_carnival
    elif "Pong" in game:
        return label_list_pong
    elif "SpaceInvaders" in game:
        return label_list_space_invaders
    else:
        raise ValueError(f"Game {game} could not be found in labels")


def get_labels(gt_bbs, game, boxes_batch):
    """
    Compare ground truth to boxes computed by SPACE
    """
    return match_bbs(gt_bbs, boxes_batch, label_list_for(game))


def get_labels_moving(gt_bbs, game, boxes_batch):
    """
    Compare ground truth to boxes computed by SPACE
    """
    return match_bbs(gt_bbs[gt_bbs[4] == "M"], boxes_batch, label_list_for(game))


def match_bbs(gt_bbs, boxes_batch, label_list):
    labels = []
    # print(gt_bbs, boxes_batch[0])
    for bb in boxes_batch:
        label, max_iou = max(((gt_bb[5], iou(bb, gt_bb)) for gt_bb in gt_bbs.itertuples(index=False, name=None)),
                             key=lambda tup: tup[1])
        if max_iou < 0.2:
            label = "no_label"
        labels.append(label)
    return torch.LongTensor([label_list.index(label) for label in labels])


def iou(bb, gt_bb):
    """
    Works in the same vein like iou, but only compares to gt size not the union, as such that SPACE is not punished for
    using a larger box, but fitting alpha/encoding
    """
    inner_width = min(bb[3], gt_bb[3]) - max(bb[2], gt_bb[2])
    inner_height = min(bb[1], gt_bb[1]) - max(bb[0], gt_bb[0])
    if inner_width < 0 or inner_height < 0:
        return 0
    # bb_height, bb_width = bb[1] - bb[0], bb[3] - bb[2]
    gt_bb_height, gt_bb_width = gt_bb[1] - gt_bb[0], gt_bb[3] - gt_bb[2]
    intersection = inner_height * inner_width
    gt_area_not_union = (gt_bb_height * gt_bb_width)
    if gt_area_not_union:
        return intersection / gt_area_not_union
    else:
        print("Gt Area is zero", gt_area_not_union, intersection, bb, gt_bb)
        return 0


#  Deprecated in favor of IOU
def labels_for_batch(boxes_batch, entity_list, row, label_list, pieces=None):
    if pieces is None:
        pieces = {}
    bbs = (boxes_batch[:, :4] * (210, 210, 160, 160)).round().astype(int)
    for en in entity_list:
        if (f'{en}_visible' not in row or row[f'{en}_visible'].item()) and f'{en}_y' in row and f'{en}_x' in row:
            pieces[en] = (row[f'{en}_y'].item(), row[f'{en}_x'].item())
    return labels_for_pieces(bbs, row, pieces, label_list)


def labels_for_pieces(bbs, row, pieces, label_list):
    labels = []
    for bb in bbs:
        label = label_for_bb(bb, row, pieces)
        labels.append(label)
    return torch.LongTensor([label_list.index(lab) for lab in labels])


def label_for_bb(bb, row, pieces):
    label = min(((name, bb_dist(bb, pos)) for name, pos in pieces.items()), key=lambda tup: tup[1])
    if label[1] > 15:  # dist
        label = ("no_label", 0)
    label = label[0]  # name
    if f'{label}_blue' in row and row[f'{label}_blue'].item():
        label = "blue_ghost"
    elif f'{label}_white' in row and row[f'{label}_white'].item():
        label = "white_ghost"
    return label


# TODO: Validate mean method with proper labels
def bb_dist(bb, pos):
    return abs(bb[0] - pos[0]) + abs(bb[2] - pos[1])