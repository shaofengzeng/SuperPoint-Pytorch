# SuperPoint-Pytorch (A Pure Pytorch Implementation)
SuperPoint: Self-Supervised Interest Point Detection and Description  


# Thanks  
This work is based on:  
- Tensorflow implementation by [Rémi Pautrat and Paul-Edouard Sarlin](https://github.com/rpautrat/SuperPoint)  
- Official [SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork). 
- [Kornia](https://kornia.github.io/)  

# Existing Problems
Different performances in training and evaluation modes.
For example, in training mode, the loss is about 2.0, while in
eval mode, the loss is about 2000!. This may be
caused by batch normalization.

# Our Performances
- MagicPoint, detection repeatability on Hpatches: 0.664
- SuperPoint, homography estimation correctness on Hpatches: 0.715
- Some Training Tricks
**a**. Remember to remove parameter eps=1e-3 for all the BatchNorma2d functions
       in model/modules/cnn/\*.py   
**b**. Set better parameters, especially for lambda_d, lambda_loss in *.yaml     
**c**. It seems that the Batch Normalization will cause the loss not converge
       (this may be the reason why magicleap didn't use BN),
       so please try to comment/uncomment the normalization in cnn_heads.py,
       vgg_backbone.py or the following lines in cnn_heads.py and loss.py, 
  
```
out_norm = torch.norm(out, p=2, dim=1)# Compute the norm.
out = out.div(torch.unsqueeze(out_norm, 1))# Divide by norm to normalize.
``` 
```
dot_product_desc = torch.reshape(F.normalize(torch.reshape(dot_product_desc, [batch_size, Hc, Wc, Hc * Wc]),
                                              p=2,
                                              dim=3), [batch_size, Hc, Wc, Hc, Wc])
dot_product_desc = torch.reshape(F.normalize(torch.reshape(dot_product_desc, [batch_size, Hc * Wc, Hc, Wc]),
                                              p=2,
                                              dim=1), [batch_size, Hc, Wc, Hc, Wc])
``` 

# New Update (09/04/2021)
* You can now reproduce [rpautrat/Superpoint](https://github.com/rpautrat/SuperPoint)'s performances with pytorch.   
* Main steps:
    - 1 Define the network by [superpoint_bn.py](model/superpoint_bn.py) (Refer to [train.py](./train.py) for more details)
    - 2 Set parameter eps=1e-3 for all the BatchNormalization functions in model/modules/cnn/*.py
    - 3 Load pretrained weight [superpoint_bn.pth](./superpoint_bn.pth) and run forward propagation
 

# Usage
* 1 Prepare your data. Make directories *data* and *export*. The data directory should look like,
    ```
    data
    |-- coco
    |  |-- train2017
    |  |     |-- a.jpg
    |  |     |-- ...
    |  --- test2017
    |        |-- b.jpg
    |        |-- ...
    |-- hpatches
    |   |-- i_ajuntament
    |   |   |--1.ppm
    |   |   |--...
    |   |   |--H_1_2
    |   |-- ...
    ```
    You can create *soft links* if you already have *coco, hpatches* data sets, the commands are,
    ```
    cd data
    ln -s dir_to_coco ./coco
    ```
* 2 The training steps are much similar to [rpautrat/Superpoint](https://github.com/rpautrat/SuperPoint). 
    However you'd better to read the scripts first so that you can give correct settings for your envs.   
    - 2.1 Train MagicPoint: `python train.py ./config/magic_point_train.yaml`   
    （Note that you have to delete the directory _./data/synthetic_shapes_ 
      whenever you want to regenerate the synthetic data set）
    - 2.2 Export coco labels: `python homo_export_labels.py #using your data dirs`
    - 2.3 Train MagicPoint on coco labels data set (exported by step 2.2)   
    `python train.py ./config/magic_point_train.py #with correct data dirs` 
    - 2.4 Train SuperPoint: `python train.py ./config/superpoint_train.py #with correct data dirs`
    - others. Validate detection repeatability:   
    ```
    python export detections_repeatability.py   
    python compute_repeatability.py
    ```  
    (NOTE: You have to edit **.yaml* files to run corresponding tasks,
     especially for the *path* or *dir* items 
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

