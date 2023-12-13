from engine.utils import get_config
from engine.train import train
from engine.multi_train import multi_train
from engine.eval import eval
from engine.show import show
from engine.data_gatherer import gather
from engine.video_splitter import split_videos
from engine.flow_example import test_flow
from engine.train_classifier import train_classifier
from engine.load_model import load_model


if __name__ == '__main__':

    task_dict = {
        'train': train,
        'eval': eval,
        'show': show,
        'split_videos': split_videos,
        'gather': gather,
        'test_flow': test_flow,
        'multi_train': multi_train,
        'train_classifier': train_classifier,
        'load_model': load_model,
    }
    cfg, task = get_config()
    assert task in task_dict
    task_dict[task](cfg)
