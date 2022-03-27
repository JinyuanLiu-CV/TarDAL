# TarDAL

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JinyuanLiu-CV/TarDAL/blob/main/tutorial.ipynb)

Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for
Object Detection.

Work have been accepted by [CVPR 2022](https://cvpr2022.thecvf.com/).

The paper and dataset will available soon.

![Abstract](assets/Firstfigure.jpg)

## Quick Start Examples

You can try our method online (free)
in [Colab](https://colab.research.google.com/github/JinyuanLiu-CV/TarDAL/blob/main/tutorial.ipynb).

### Install

We recommend you to use the conda management environment.

```shell
conda create -n tardal python=3.8
conda activate tardal
pip install -r requirements.txt
```

### Fuse or Eval

We offer three pre-trained models.

| Name     | Description                                                     |
|----------|-----------------------------------------------------------------|
| TarDAL   | Optimized for human vision. (Default)                           |
| TarDAL+  | Optimized for object detection.                                 |
| TarDAL++ | Optimal solution for joint human vision and detection accuracy. |

```shell
python fuse.py --src data/sample/s1 --dst runs/sample/tardal --weights weights/tardal.pt --color
python fuse.py --src data/sample/s1 --dst runs/sample/tardal+ --weights weights/tardal+.pt --color --eval
python fuse.py --src data/sample/s1 --dst runs/sample/tardal++ --weights weights/tardal++.pt --color --eval
```

> `--color` will colorize the fused images with corresponding visible color space.
