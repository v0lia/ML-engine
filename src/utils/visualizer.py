# visualizer.py

import copy
import numpy as np
from math import sqrt, ceil

#import matplotlib.pyplot as plt

import torch
import torchvision.utils

from src.data import dataset_utils

def add_model_graph(model, image, writer):
    model_cpu = copy.deepcopy(model).to('cpu')
    image_cpu = image.to('cpu')
    writer.add_graph(model_cpu, image_cpu)
    return

def add_embedding(dataloader, writer, n=128, classes=dataset_utils.classes, tag="Data embedding"):
    n = min(n, len(dataloader.dataset))
    images, labels = dataloader.dataset.data, dataloader.dataset.targets
    perm = torch.randperm(len(images))[:n]   # permutation
    images, labels = images[perm][:n], labels[perm][:n]
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    images = dataset_utils.normalize_images(images)
    features = images.reshape(n, -1)   # flatten
    class_labels = [classes[int(l)] for l in labels]
    label_img = dataset_utils.get_label_img(images.to('cpu'))
    writer.add_embedding(features.detach().to('cpu'),
                    metadata=class_labels,
                    label_img=label_img,
                    tag=tag)

def add_scalar(writer, tag, value, step=None):
    writer.add_scalar(tag, value, step)
    return 
    
def add_pr_curve(class_index, test_probs, test_labels, writer, step=None):
    pr_curve_probs = test_probs[:, class_index]
    pr_curve_truth = test_labels == class_index

    writer.add_pr_curve(dataset_utils.classes[class_index],
                        pr_curve_truth.detach().to('cpu'),
                        pr_curve_probs.detach().to('cpu'),
                        global_step=step)
    return

def add_pr_curves(test_probs, test_labels, writer, step=None):
    for class_index in range(len(dataset_utils.classes)):
        add_pr_curve(class_index, test_probs, test_labels, writer, step=step)

def resolve_grid_scale(nrow, batch_size):
    n_images = min(nrow**2, batch_size)
    nrow = ceil(sqrt(n_images)) if n_images < nrow**2 else nrow
    return nrow, n_images

def add_sample_grid(images, writer, step=None, nrow=3, tag="Sample grid"):
    nrow, n_images = resolve_grid_scale(nrow, images.size(0))
    images = images[:n_images]
 
    img_grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)   # , scale_each=True for better visibility   # , pad_value=0.5 for visible grid borders
    writer.add_image(tag, img_grid.detach().to('cpu'), step)
    return

# Someday
'''def add_prediction_grid(model, images, labels, writer, step, nrow=3, tag="Prediction grid"):
    add_sample_grid(images, writer, step=step, nrow=nrow, tag=tag)
    add_prediction_labels_grid(model, images, labels, writer, nrow=nrow, step=step, class_names=dataset_utils.classes)
    return

def add_prediction_labels_grid(model, images, labels, writer, step, nrow=3, tag="Prediction labels grid", class_names=None, label_font=12, label_padding=2):
    nrow, n_images = resolve_grid_scale(nrow, images.size(0))
    images, labels = images[:n_images], labels[:n_images]
    model.eval() 
    with torch.no_grad():
          preds = model(images).argmax(dim=1)
   
    # TBD

    writer.add_figure(tag=tag, figure=fig, step=step)
    return '''  

