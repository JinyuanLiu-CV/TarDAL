# README

This document is intended to assist users in reproducing the experiments described in the paper.

## Experiments

In this section, we'll go over the steps involved in reproducing the figures in our paper.

### Figure 4

```shell
python fuse.py --src data/sample/s1 --dst runs/sample/s1 --weights weights/default/tardal.pt --color
```

This command will assist you in generating the fused results of the three pairs of images indicated in Figure 4.
[s1](runs/sample/s1) is where the fused images will be saved.

### Figure 6

```shell
python fuse.py --src data/sample/s2 --dst runs/sample/s2 --weights weights/default/tardal.pt --color
```

This command will assist you in generating the fused results of the two pairs of images indicated in Figure 6.
[s1](runs/sample/s2) is where the fused images will be saved.

We have placed the results of the yolo in the folder, and you can use the following command to draw the detection
bounding box over the fused results.

```shell
python utils/draw_bounding.py --img runs/sample/s2 --label data/sample/s2/label
```

This will overwrite the raw fused images with the marked fused images. You can also use our trained
yolo [models](data/sample/s2/weights) to repredict the labels, with this way, you need to download yolov5
from [github](https://github.com/ultralytics/yolov5) and following the documents provided by yolov5 teams.
The [a1.pt](data/sample/s2/weights/a1.pt) for MS detection(8-classes) and the [a2.pt](data/sample/s2/weights/a2.pt) for
M3FD detection(5-classes). The args `--save-txt --save-conf` should be added during the validation process.

### Figure 7

```shell
python fuse.py --src data/sample/s3 --dst runs/sample/s3/{x} --weights weights/ablation/{x}.pt --color
```

The {x} could be {m1, m2, m3, m4}, corresponding to {base network,without DT, without DD, full model}, respectively. (
The m4 is the same as our [default](weights/default/tarbal.pt) model.)

This command will assist you in generating the fused results of the three pairs of images indicated in Figure 7.
[s3](runs/sample/s3) is where the fused images will be saved.

### Figure 8

```shell
python fuse.py --src data/sample/s4 --dst runs/sample/s4/{x} --weights weights/ablation/{x}.pt --color
```

The {x} could be {m5, m6}, corresponding to {without SDW, without m}, respectively.

This command will assist you in generating the fused results of the three pairs of images indicated in Figure 8.
[s4](runs/sample/s4) is where the fused images will be saved.

### Figure 9

In a similar way to generation in Figure 5, you can use the following code to generate the fused images and use tools we
provided to draw the bounding boxes on images.

```shell
python fuse.py --src data/sample/s5 --dst runs/sample/s5/e1 --weights weights/ablation/e1.pt --color --eval
python fuse.py --src data/sample/s5 --dst runs/sample/s5/e2 --weights weights/ablation/e2.pt --color --eval
python fuse.py --src data/sample/s5 --dst runs/sample/s5/e3 --weights weights/ablation/e3.pt --color
```

The {e1, e2, e3} correspond to {JT, TT, DT}, respectively. (The e3 is the same as
our [default](weights/default/tarbal.pt) model.)

```shell
python utils/draw_bounding.py --img runs/sample/s5/{x} --label data/sample/s5/label/{x}
```

The {x} could be {e1, e2, e3}, and this will overwrite the raw fused images with the marked fused images.
