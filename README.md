# MFuseNet

This is the official implementation code for MFuseNet. For technical details, please refer to :

**MFuseNet: Robust Depth Estimation with Learned Multiscopic Fusion** <br />
[Weihao Yuan](https://weihao-yuan.com), Rui Fan, Michael Yu Wang, Qifeng Chen <br />
**ICRA2020, RA-L** <br />
**[[Paper](https://ieeexplore.ieee.org/document/9000612)] [[Project Page](https://sites.google.com/view/multiscopic)]** <br />


<div align="center">
<img src="http://weihao-yuan.com/wp-content/uploads/2019/05/camera.jpg" width="300px" />
</div>

### Bibtex
If you find this code useful, please consider citing:

```
@article{yuan2020mfusenet,
  title={MFuseNet: Robust Depth Estimation with Learned Multiscopic Fusion},
  author={Yuan, Weihao and Fan, Rui and Wang, Michael Yu and Chen, Qifeng},
  journal={IEEE Robotics and Automation Letters},
  year={2020},
  publisher={IEEE}
}
```


## Contents
1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Train](#train)


## Environment setup
This code has been tested on Ubuntu 16.04, CUDA 9.0, two GTX 1080 Ti GPUs.

**Dependencies**:

- Python2.7
- PyTorch (0.4.0+)
- torchvision (0.2.0+)
- os, time, numpy, argparse, cv2, matplotlib, PIL


## Data Preparation
The input of the network are the cost volumes obtained by cost calculation step in stereo matching algorithms. They can be calculated by block matching, semi-global matching, graph cuts, deep-network-based methods, etc. The default costs are obtained by MC-CNN. Please refer to [MC-CNN](https://github.com/jzbontar/mc-cnn) for computing the cost volumes. 

The training data for three-view fusion are organized as follows:
```
dataset/
    TRAIN/
        scene1/
            view0.png
            view1.png
            view2.png
            disp1.png
            left.bin
            right.bin
    TEST/
    EVAL/

```
The `view0.png`, `view1.png`, `view2.png` are the color images of the left, center, and right view. The `disp1.png` is the ground-truth disparity map for view1. The `left.bin` and `right.bin` are the cost volumes obtained by MC-CNN for the matching between the left, right view and the center view.

For five-view fusion, there are additional `view3.png` for the bottom view and `view4.png` for the top view, and their corresponding cost volumes `bottom.bin` and `top.bin`.

[Example data](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wyuanaa_connect_ust_hk/EhQpE6ypGlpKmdHTpJtZA_YBjvIyjVSEWJPifRvr2THmLQ?e=QNoVkf) are available here.


## Train:

```
. train.sh
```





## License
Licensed under an MIT license.