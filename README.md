# TarDAL 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JinyuanLiu-CV/TarDAL/blob/main/tutorial.ipynb)
![visitors](https://visitor-badge.glitch.me/badge?page_id=JinyuanLiu-CV.TarDAL)

Jinyuan Liu, Xin Fan*, Zhangbo Huang, Guanyao Wu, Risheng Liu , Wei Zhong, Zhongxuan Luo,**“Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection”**, IEEE/CVF Conference on Computer Vision and Pattern Recognition **(CVPR)**, 2022. **(Oral)**


- [*[ArXiv]*](https://arxiv.org/abs/2203.16220v1)
---

![Abstract](assets/first_figure.jpg)

---


<h2> <p align="center"> M3FD Dataset </p> </h2>  

### Preview
The preview of our dataset is as follows.

---

![preview](assets/Preview.png)
![gif1](assets/Preview.gif)
 
---

### Details
- **Sensor**: A synchronized system containing one binocular optical camera and one binocular infrared sensor. More details are available in the paper.

- **Main scene**: 
   - Campus of Dalian University of Technology.
   - State Tourism Holiday Resort at the Golden Stone Beach in Dalian, China.
   - Main roads in Jinzhou District, Dalian, China.

- **Total number of images**: 
   - **8400** (for fusion, detection and fused-based detection)
   - **600** (independent scene for fusion)

- **Total number of image pairs**:
   - **4200** (for fusion, detection and fused-based detection)
   - **300** (independent scene for fusion)


- **Format of images**: 
   - [Infrared] 24-bit grayscale bitmap
   - [Visible]  24-bit color bitmap

- **Image size**: **1024 x 768** pixels (mostly)

- **Registration**: **All image pairs are registered.** The visible images are calibrated by using the internal parameters of our synchronized system, and the infrared images are artificially distorted by homography matrix.

- **Labeling**: **34407 labels** have been manually labeled, containing 6 kinds of targets: **{People, Car, Bus, Motorcycle, Lamp, Truck}**. (Limited by manpower, some targets may be mismarked or missed. We would appreciate if you would point out wrong or missing labels to help us improve the dataset)

### Download
   - [Google Drive](https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6?usp=sharing)
   - [Baidu Yun](https://pan.baidu.com/s/1GoJrrl_mn2HNQVDSUdPCrw?pwd=M3FD)
### File structure
```
  M3FD
  ├── Challenge
  |   ├── Beach
  |   |   ├──Annotation
  |   |   |  ├── 01863.xml
  |   |   |  └── ...
  |   |   ├──Ir
  |   |   |  ├── 01863.png
  |   |   |  └── ...
  |   |   ├──Vis
  |   |   |  ├── 01863.png
  |   |   |  └── ...
  |   ├── Crossroads
  |   └── ...
  ├── Daytime
  |   ├── Alley
  |   └── ...
  ├── Night
  |   ├── Basement
  |   └── ...
  └── Overcast
      ├── Atrium
      └── ...
```
If you have any question or suggestion about the dataset, please email to [Guanyao Wu](mailto:rollingplainko@gmail.com) or [Jinyuan Liu](mailto:atlantis918@hotmail.com).

<h2> <p align="center"> TarDAL Fusion </p> </h2>  

### Baselines(Sorted alphabetically)
   - [AUIF](https://ieeexplore.ieee.org/document/9416456) (IEEE TCSVT 2021)
   - [DDcGAN](https://github.com/hanna-xu/DDcGAN) (IJCAI 2019)
   - [Densefuse](https://github.com/hli1221/imagefusion_densefuse) (IEEE TIP 2019)
   - [DIDFuse](https://github.com/Zhaozixiang1228/IVIF-DIDFuse) (IJCAI 2020)
   - [FusionGAN](https://github.com/jiayi-ma/FusionGAN) (Information Fusion 2019)
   - [GANMcC](https://github.com/HaoZhang1018/GANMcC) (IEEE TIM 2021)
   - [MFEIF](https://github.com/JinyuanLiu-CV/MFEIF) (IEEE TCSVT 2021)
   - [RFN-Nest](https://github.com/hli1221/imagefusion-rfn-nest) (Information Fusion 2021)
   - [SDNet](https://github.com/HaoZhang1018/SDNet) (IJCV 2021)
   - [U2Fusion](https://github.com/hanna-xu/U2Fusion) (IEEE TPAMI 2020)


### Fuse Quick Start Examples

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


If you have any question about the code, please email to [Zhanbo Huang](mailto:zbhuang@mail.dlut.edu.cn).

## Citation
```
@inproceedings{TarDAL,
  title={Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection},
  author={Jinyuan Liu, Xin Fan*, Zhangbo Huang, Guanyao Wu, Risheng Liu , Wei Zhong, Zhongxuan Luo},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
