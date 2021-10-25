from detectron2.config import CfgNode as CN


def add_norcal_config(cfg):
    cfg.TEST.CALIBRATION = CN()
    cfg.TEST.CALIBRATION.METHOD = "gamma"
    cfg.TEST.CALIBRATION.WITH = "exps"
    cfg.TEST.CALIBRATION.GAMMA = 0.0  # Nc^gamma
    cfg.TEST.CALIBRATION.RENORM = True
