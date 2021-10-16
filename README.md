# SuperPoint-Pytorch (A Pure Pytorch Implementation)
SuperPoint: Self-Supervised Interest Point Detection and Description  

# Thanks  
This work is based on:  
- Tensorflow implementation by [Rémi Pautrat and Paul-Edouard Sarlin](https://github.com/rpautrat/SuperPoint)  
- Official [SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork). 
- [Kornia](https://kornia.github.io/)

# New update (20211016)
- Train your MagicPoint and SuperPoint


# New update (20210904)
* You can now reproduce [rpautrat/Superpoint](https://github.com/rpautrat/SuperPoint) with pytorch.   
* Main Steps:
    - 1 Define network by [superpoint_bn.py](model/superpoint_bn.py) (Refer to [train.py](./train.py) for more details)
    - 2 Set parameter eps=1e-3 for all BatchNormalization functions
    - 3 Load pretrained weight [superpoint_bn.pth](./superpoint_bn.pth) and run forward propagation
 

# Usage
* 1 Prepare your data. Make directories *data* and *export*. The data directory should look like,
    ```
    data
    |-- coco
    |  |-- train2017
    |  |     |-- a.jpg
    |  |     |-- ...
    |  `-- test2017
    |        |-- b.jpg
    |        |-- ...
    |-- hpatches
    |   |-- i_ajuntament
    |   |   |--1.ppm
    |   |   |--...
    |   |   |--H_1_2
    |   |-- ...
    ```
    You can create *soft links* if you already have *coco, hpatches* data sets, the command is,
    ```
    cd data
    ln -s dir_to_coco ./coco
    ```
* 2 The training steps are much similar to [rpautrat/Superpoint](https://github.com/rpautrat/SuperPoint)  
    - 2.1 Train MagicPoint: `python train.py ./config/magic_point_train.yaml`
    - 2.2 Export coco labels: `python export detections.py`
    - 2.3 Train MagicPoint on coco labels data set (export by step 2.2)   
    `python train.py ./config/superpoint_train.py`   
    - 2.4 Train SuperPoint.py: `python train.py ./config/superpoint_train.py`
    - others. Validate detection repeatability:   
    ```
    python export detections_repeatability.py   
    python compute_repeatability.py
    ```  
    (NOTE: You have to edit **.yaml* files to run corresponding tasks, especially for the following items  
    ```
    model
        name: superpoint # magicpoint
     ...
    data:
        name: coco #synthetic
        image_train_path: ['./data/mp_coco_v2/images/train2017',] #several data sets can be list here
        label_train_path: ['./data/mp_coco_v2/labels/train2017/',]
        image_test_path: './data/mp_coco_v2/images/test2017/'
        label_test_path: './data/mp_coco_v2/labels/test2017/'
    ```

