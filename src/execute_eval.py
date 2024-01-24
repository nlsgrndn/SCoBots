from engine.eval_model_and_classifier import eval_model_and_classifier
from engine.eval_classifier import eval_classifier
from engine.train_classifier import train_classifier
from engine.utils import get_config_v2
import numpy as np
from engine.eval import eval

models_path = "../tmp_models"
cfg_path_for_game = {
    "pong": "configs/my_atari_pong_gpu.yaml",
	"boxing": "configs/my_atari_boxing_gpu.yaml",
	"skiing": "configs/my_atari_skiing_gpu.yaml",
}
seeds = np.arange(5, 10)
for game in cfg_path_for_game.keys():
    cfg = get_config_v2(cfg_path_for_game[game])
    for seed in seeds:
        cfg_override_list = [
            "exp_name", f"{game}_seed{seed}",
            "resume_ckpt", f"{models_path}/{game}_seed{seed}/model_000005001.pth",
            "logdir", f"{models_path}",
        ]
        
        cfg.merge_from_list(cfg_override_list)
        eval(cfg)
        train_classifier(cfg)
        eval_classifier(cfg)
        eval_model_and_classifier(cfg)
        print("one iteration done")





