__all__ = ['get_evaluator']

from .space_eval import SpaceEval

def get_evaluator(cfg):
    return SpaceEval(cfg)


