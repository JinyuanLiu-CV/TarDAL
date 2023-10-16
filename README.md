# TarDAL 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JinyuanLiu-CV/TarDAL/blob/main/tutorial.ipynb)
![visitors](https://visitor-badge.glitch.me/badge?page_id=JinyuanLiu-CV.TarDAL)

Jinyuan Liu, Xin Fan*, Zhangbo Huang, Guanyao Wu, Risheng Liu , Wei Zhong, Zhongxuan Luo,**“Target-aware Dual
Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection”**,
IEEE/CVF Conference on Computer Vision and Pattern Recognition **(CVPR)**, 2022. **(Oral)**

- [*[ArXiv]*](https://arxiv.org/abs/2203.16220v1)
- [*[CVPR]*](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Target-Aware_Dual_Adversarial_Learning_and_a_Multi-Scenario_Multi-Modality_Benchmark_To_CVPR_2022_paper.pdf)

---

![Abstract](assets/first_figure.jpg)

---


<h2> <p align="center"> M3FD Dataset </p> </h2>  

### Preview

The preview of our dataset is as follows.

---

![preview](assets/preview.png)
![gif](assets/preview.gif)
 
---

### Details

- **Sensor**: A synchronized system containing one binocular optical camera and one binocular infrared sensor. More
  details are available in the paper.

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

- **Registration**: **All image pairs are registered.** The visible images are calibrated by using the internal
  parameters of our synchronized system, and the infrared images are artificially distorted by homography matrix.

- **Labeling**: **34407 labels** have been manually labeled, containing 6 kinds of targets: **{People, Car, Bus,
  Motorcycle, Lamp, Truck}**. (Limited by manpower, some targets may be mismarked or missed. We would appreciate if you
  would point out wrong or missing labels to help us improve the dataset)

### Download

- [Google Drive](https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6?usp=sharing)
- [Baidu Yun](https://pan.baidu.com/s/1GoJrrl_mn2HNQVDSUdPCrw?pwd=M3FD)


If you have any question or suggestion about the dataset, please email to [Guanyao Wu](mailto:rollingplainko@gmail.com)
or [Jinyuan Liu](mailto:atlantis918@hotmail.com).

<h2> <p align="center"> TarDAL Fusion </p> </h2>  

### Baselines

In the experiment process, we used the following **outstanding** work as our baseline.

*Note: Sorted alphabetically*

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

### Quick Start

Under normal circumstances, you may just be curious about the results of the fusion task, so we have prepared an online demonstration.

Our online preview (free) in [Colab](https://colab.research.google.com/github/JinyuanLiu-CV/TarDAL/blob/main/tutorial.ipynb).

### Set Up on Your Own Machine

When you want to dive deeper or apply it on a larger scale, you can configure our TarDAL on your computer following the steps below.

#### Virtual Environment

We strongly recommend that you use Conda as a package manager.

```shell
# create virtual environment
conda create -n tardal python=3.10
conda activate tardal
# select pytorch version yourself
# install tardal requirements
pip install -r requirements.txt
# install yolov5 requirements
pip install -r module/detect/requirements.txt
```

#### Data Preparation

You should put the data in the correct place in the following form.

```
TarDAL ROOT
├── data
|   ├── m3fd
|   |   ├── ir # infrared images
|   |   ├── vi # visible images
|   |   ├── labels # labels in txt format (yolo format)
|   |   └── meta # meta data, includes: pred.txt, train.txt, val.txt
|   ├── tno
|   |   ├── ir # infrared images
|   |   ├── vi # visible images
|   |   └── meta # meta data, includes: pred.txt, train.txt, val.txt
|   ├── roadscene
|   └── ...
```

You can directly download the TNO and RoadScene datasets organized in this format from here.

- [Google Drive](https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6?usp=sharing)
- [Baidu Yun](https://pan.baidu.com/s/1GoJrrl_mn2HNQVDSUdPCrw?pwd=M3FD)

#### Fuse or Eval

In this section, we will guide you to generate fusion images using our pre-trained model.

As we mentioned in our paper, we provide three pre-trained models.

| Name      | Description                                                     |
|-----------|-----------------------------------------------------------------|
| TarDAL-DT | Optimized for human vision. (Default)                           |
| TarDAL-TT | Optimized for object detection.                                 |
| TarDAL-CT | Optimal solution for joint human vision and detection accuracy. |

You can find their corresponding configuration file path in [configs](config/official/infer).

Some settings you should pay attention to:

* config.yaml
    * `strategy`: save images (fuse) or save images & labels (fuse & detect)
    * `dataset`: name & root
    * `inference`: each item in inference
* infer.py
    * `--cfg`: config file path, such as `configs/official/tardal-dt.yaml`
    * `--save_dir`: result save folder

Under normal circumstances, you don't need to manually download the model parameters, our program will do it for you.

```shell
# TarDAL-DT
# use official tardal-dt infer config and save images to runs/tardal-dt
python infer.py --cfg configs/official/tardal-dt.yaml --save_dir runs/tardal-dt
# TarDAL-TT
# use official tardal-tt infer config and save images to runs/tardal-tt
python infer.py --cfg configs/official/tardal-tt.yaml --save_dir runs/tardal-tt
# TarDAL-CT
# use official tardal-ct infer config and save images to runs/tardal-ct
python infer.py --cfg configs/official/tardal-ct.yaml --save_dir runs/tardal-ct
```

#### Train

We provide some training script for you to train your own model.

Please note: The training code is only intended to assist in understanding the paper and is not recommended for direct application in
production environments.

Unlike previous code versions, you don't need to preprocess the data, we will automatically calculate the IQA weights and mask.

```shell
# TarDAL-DT
python train.py --cfg configs/official/tardal-dt.yaml --auth $YOUR_WANDB_KEY
# TarDAL-TT
python train.py --cfg configs/official/tardal-tt.yaml --auth $YOUR_WANDB_KEY
# TarDAL-CT
python train.py --cfg configs/official/tardal-ct.yaml --auth $YOUR_WANDB_KEY
```

If you want to base your approach on ours and extend it to a production environment, here are some additional suggestions for you.

[Suggestion: A better train process for everyone.](assets/train_process.png)

### Any Question

If you have any other questions about the code, please email [Zhanbo Huang](mailto:zbhuang917@hotmail.com).

Due to job changes, the previous link `zbhuang@mail.dlut.edu.cn` is no longer available.

## Citation

If this work has been helpful to you, please feel free to cite our paper!

```
@inproceedings{liu2022target,
  title={Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection},
  author={Liu, Jinyuan and Fan, Xin and Huang, Zhanbo and Wu, Guanyao and Liu, Risheng and Zhong, Wei and Luo, Zhongxuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5802--5811},
  year={2022}
}
```
