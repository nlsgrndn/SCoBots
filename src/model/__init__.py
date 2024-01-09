from .space.space import Space
#from .low_res_space.time_consistency import LrTcSpace

__all__ = ['get_model']


def get_model(cfg):
    """
    :param cfg:
    :return:
    """
    model = None
    #removed
    #if cfg.model.lower() in ['lrspace', "lrtcspace", "tclrspace"]:
    #    model = LrTcSpace() 
    if cfg.model.lower() == 'space':
        model = Space()
    return model
