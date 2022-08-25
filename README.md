# VizWiz-FewShot

![VizWiz-FewShot Cover Image](https://vizwiz.s3.us-east-2.amazonaws.com/cover.png "VizWiz-FewShot")

# Introduction

The VizWiz-FewShot dataset provides nearly 10,000 segmentations of 100 categories on over 4,500 images that were taken by people with visual impairments. These annotations include unique features compared to other datasets, such as holes in objects, a larger range in object size, and significantly more objects containing text. Please read our paper to learn more:

[VizWiz-FewShot: Locating Objects in Images Taken by People With Visual Impairments.
Yu-Yun Tseng, Alexander Bell, and Danna Gurari. European Conference on Computer Vision (ECCV), 2022.](https://arxiv.org/abs/2207.11810)

You are in the right place if you are looking for the VizWiz-FewShot API. This API aims to make it as easy as possible to train and evaluate your *object detection* or *instance segmentation* model on the VizWiz-FewShot dataset. Read on for more information.

## Dataset download

Before you can use our API, you must download the dataset. Links to download the images and annotations are available [here](https://vizwiz.org/tasks-and-datasets/object-localization/).

Alternatively, you can run the following series of commands. This will create a directory `dataset` with all the files in your current directory.
```
$ mkdir -p dataset/images dataset/annotations
$ cd dataset
$ wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip \
       https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip \
       https://vizwiz.s3.us-east-2.amazonaws.com/annotations.zip \
       https://vizwiz.s3.us-east-2.amazonaws.com/annotations.json
$ unzip -o train.zip -d images
$ unzip -o val.zip -d images
$ unzip -o annotations.zip -d annotations
$ rm train.zip val.zip annotations.zip
```

# API

## Installation

Please run the following command to install our Python package from PyPI.
```
$ pip install vizwiz-fewshot
```

## PyTorch usage

We provide implementations of PyTorch's `torch.utils.data.Dataset` for both object detection and instance segmentation. You will supply an instance to a `torch.utils.data.DataLoader`. To read more about how these work on a deeper level, navigate [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

**Note: This assumes you are following the standard *4-fold cross-validation* approach for model evaluation used in the few-shot learning community.**

### Object detection

```python
from vizwiz import ObjectDetectionDataset
...

# Initialize Dataset's

IMAGE_PATH = 'dataset/images'
ANNOTATION_PATH = 'dataset/annotations.json'

# You may need special transforms on the images for your data.
# There is no transformation by default. However, here we
# transform the images into a `torch.tensor` object.
base_dataset = ObjectDetectionDataset(
    root=IMAGE_PATH,
    annFile=ANNOTATION_PATH,
    transform=torchvision.transforms.ToTensor(),
    fold=0, # Enter the fold here,
    set_type='base'
)

support_dataset = ObjectDetectionDataset(
    root=IMAGE_PATH,
    annFile=ANNOTATION_PATH,
    transform=torchvision.transforms.ToTensor(),
    fold=0,
    set_type='support',
    shots=1 # Enter the number of shots here
)

query_dataset = ObjectDetectionDataset(
    root=IMAGE_PATH,
    annFile=ANNOTATION_PATH,
    transform=torchvision.transforms.ToTensor(),
    fold=0,
    set_type='query'
)

# Initialize DataLoader's. Configure as needed following the PyTorch docs.
base_loader = torch.utils.data.DataLoader(base_dataset)
support_loader = torch.utils.data.DataLoader(support_dataset)
query_loader = torch.utils.data.DataLoader(query_dataset)

# Training

# Begin by training on the base set
for i, data in enumerate(base_loader):
    ...

# Fine-tune on the support set
for i, data in enumerate(support_loader):
    ...

# Evaluate on the query set
for i, data in enumerate(query_dataset):
    ...

...
```

### Instance segmentation

*Instructions will be posted soon.*

# Citation

If you make use of our dataset for your research, please be sure to cite our work with the following BibTeX citation.
```
@misc{https://doi.org/10.48550/arxiv.2207.11810,
  doi = {10.48550/ARXIV.2207.11810},
  url = {https://arxiv.org/abs/2207.11810},
  author = {Tseng, Yu-Yun and Bell, Alexander and Gurari, Danna},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {VizWiz-FewShot: Locating Objects in Images Taken by People With Visual Impairments},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
