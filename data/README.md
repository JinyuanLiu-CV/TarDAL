# Dataset Configure Reference 

## Official Supported Datasets

* TNO: fuse
* RoadScene: fuse
* MultiSpectral: fuse + detect
* M3FD: fuse + detect

## Other Datasets

You can write scripts for your own custom dataset in `loader/{$NAME}.py`, and raise a pull request (optional).

## Prepare

Datasets should have the following structure:

```
data
|__ TNO // name of the dataset
    |__ ir // infrared images
    |__ vi // visible images
    |__ meta // dataset meta information
        |__ train.txt // image name for training
        |__ val.txt // image name for validation
|__ M3FD // name of the dataset
    |__ ir // infrared images
    |__ vi // visible images
    |__ labels // object labels (ground truth, cxcywh)
    |__ meta // dataset meta information
        |__ train.txt // image name for training
        |__ val.txt // image name for validation
```
