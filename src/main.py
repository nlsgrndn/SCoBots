from engine.utils import get_config
from engine.train import train
from engine.multi_train import multi_train
from engine.eval import eval
from engine.train_classifier import train_classifier
from create_latent_dataset import create_latent_dataset
from motrackers.script import execute as motrackers_main
from engine.eval_model_and_classifier import eval_model_and_classifier
from engine.eval_classifier import eval_classifier



if __name__ == '__main__':

    task_dict = {
        'train': train,
        'eval': eval,
        'multi_train': multi_train,
        'train_classifier': train_classifier,
        'eval_classifier': eval_classifier,
        'eval_model_and_classifier': eval_model_and_classifier,
        'create_latent_dataset': create_latent_dataset,
        'motrackers': motrackers_main,
    }
    cfg, task = get_config()
    assert task in task_dict
    task_dict[task](cfg)
