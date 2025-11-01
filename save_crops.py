""" 
    This python script is used to detect face from a source folder path and save the crops to a destination folder path 
    Usage: 
        1. Specify the source and destination folders in the main() function
        2. run this .py file
"""

from pathlib import Path
import argparse
import torch

import detect

def main():
    """change the source and crops_dir attribute to desired file path"""
    save_crops(
        source=r"D:\Programming\ProgrammingProjects\humanoid-classifier\2d_anime_dataset_extra",
        crops_dir=r"D:\Programming\ProgrammingProjects\humanoid-classifier\2d_anime_dataset_cropped"
    )
    return

def save_crops(source: str,
               weights: str = 'weights/yolov5s_anime.pt',
               crops_dir: str | None = None,
               output: str | None = None,
               img_size: int = 640,
               conf_thres: float = 0.4,
               iou_thres: float = 0.5,
               device: str = '',
               classes: list[int] | None = None,
               view_img: bool = False):
    """Run detect.detect() programmatically with a custom crops dir.

    If crops_dir is provided it will be used as the destination for per-class
    crop folders. If omitted, the script will save crops under `--output` as
    before.
    """

    ns = argparse.Namespace()
    # match attribute names used by detect.detect
    ns.weights = [weights]
    ns.source = source
    ns.output = output or 'inference/output'
    ns.img_size = img_size
    ns.conf_thres = conf_thres
    ns.iou_thres = iou_thres
    ns.device = device
    ns.view_img = view_img
    ns.save_txt = False
    ns.classes = classes
    # detect.py uses nargs='?' for save_crops: set to provided path or '__DEFAULT__'
    ns.save_crops = str(crops_dir) if crops_dir else '__DEFAULT__'
    ns.agnostic_nms = False
    ns.augment = False
    ns.update = False

    # assign to detect module global expected by detect.detect
    detect.opt = ns

    # run
    with torch.no_grad():
        detect.detect()


if __name__ == '__main__':
    main()