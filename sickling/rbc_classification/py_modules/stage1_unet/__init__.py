"""Stage 1 U-Net — frozen at inference time. The training loop lives in
``training 2.ipynb``; this package only provides the architecture + loader +
sliding-window predictor needed to use the trained weights downstream.
"""
from sickling.rbc_classification.py_modules.stage1_unet.bulk import bulk_predict
from sickling.rbc_classification.py_modules.stage1_unet.inference import UNet, load_unet, predict_label_map

__all__ = ["UNet", "bulk_predict", "load_unet", "predict_label_map"]
