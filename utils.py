import numpy as np
import torch
from PIL import Image , ImageSequence
from tqdm import tqdm 


def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)

def show_tiff(tiff_file, save_path):
    im = Image.open(tiff_file)
    for i, page in enumerate(tqdm(ImageSequence.Iterator(im))):
        page.save(f"{save_path}/slice%d.png" % i)


def miou(outputs, labels):
    outputs = outputs.byte()
    labels = labels.byte()
    intersection = torch.logical_and(labels, outputs)
    union = torch.logical_or(labels, outputs)
    iou_score = torch.sum(intersection) / torch.sum(union)

    if torch.isnan(iou_score.mean()):
        return torch.tensor(1)
    return iou_score.mean()