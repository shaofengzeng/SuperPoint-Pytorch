# SuperPoint-Pytorch (A Pure Pytorch Implementation)
SuperPoint: Self-Supervised Interest Point Detection and Description  


# Thanks  
This work is based on:  
- Tensorflow implementation by [Rémi Pautrat and Paul-Edouard Sarlin](https://github.com/rpautrat/SuperPoint)  
- Official [SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork).
- [pytorch-superpoint](https://github.com/eric-yyjau/pytorch-superpoint) 
- [Kornia](https://kornia.github.io/)  

# Training suggestions
A good magicpoint which can generate repeatable keypoints is important for superpoint 


# Result
__Note that the results of superpoint (rpautrat) are copied from [here] (https://github.com/rpautrat/SuperPoint/tree/master). However our model is trained on coco 2017 test set__

### Detector evaluation
Repeatability on HPatches computed with 300 points detected in common between pairs of images and with a NMS of 4:
 <table style="width:100%">
  <tr>
    <th></th>
    <th>Illumination</th>
    <th>Viewpoint</th>
  </tr>
  <tr>
    <td>Superpoint (rpautrat)</td>
    <td>0.662</td>
    <td>0.674</td>
  </tr>
  <tr>
    <td>Our</td>
    <td>0.653</td>
    <td>0.651</td>
  </tr>
</table>

 ### Descriptors evaluation
Homography estimation on HPatches computed with a maximum of 1000 points detected in common between pairs of images, a threshold of correctness of 3 and with a NMS of 8:
 <table style="width:100%">
  <tr>
    <th></th>
    <th>Illumination</th>
    <th>Viewpoint</th>
  </tr>
  <tr>
    <td>SuperPoint (rpautrat)</td>
    <td>0.965</td>
    <td>0.712</td>
  </tr>
  <tr>
    <td>Our</td>
    <td>0.933</td>
    <td>0.688</td>
  </tr>
</table>
 






# Usage
* 0 Update your repository to the latested version (if you have pulled it before)
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
    |   |-- i_ajustment
    |   |   |--1.ppm
    |   |   |--...
    |   |   |--H_1_2
    |   |-- ...
    ```
    Create *soft links* if you already have *coco, hpatches* data sets, commands are like,
    ```
    cd data
    ln -s dir_to_coco ./coco
    ```
* 2 Training steps are much similar to [rpautrat/Superpoint](https://github.com/rpautrat/SuperPoint). 
    **However we strongly suggest you read the scripts first before training**
    - 2.0 Modify the following code in train.py, to save your models, if necessary  
          `if (i%118300==0 and i!=0) or (i+1)==len(dataloader['train']):`  
    - 2.1 set proper values for training epoch in _*.yaml_.
    - 2.2 Train MagicPoint (>1 hours):  
          `python train.py ./config/magic_point_syn_train.yaml`   
          (Note that you have to delete the directory _./data/synthetic_shapes_ whenever you want to regenerate it)
    - 2.3 Export *coco labels data set v1* (>50 hours):   
          `python homo_export_labels.py #run with your data path`
    - 2.4 Train MagicPoint on *coco labels data set v1* (exported by step 2.2)       
          `python train.py ./config/magic_point_coco_train.yaml #run with your data path` 
    - 2.5 Export *coco labels data set v2* using the magicpoint trained by step 2.3
    - 2.6 Train SuperPoint using *coco labels data set v2* (>12 hours)    
          `python train.py ./config/superpoint_train.yaml #run with your data path`  
    - others. Validate detection repeatability or description  
        ```
        python export_detections_repeatability.py #(very fast)  
        python compute_repeatability.py  #(very fast)
        ## or
        python export_descriptors.py #(> 5.5 hours) 
        python compute_desc_eval.py #(> 1.5 hours)
        ```   
        
    **Descriptions of some important hyper-parameters in YAML files**
    ```
    model
        name: superpoint # superpoint or magicpoint
        pretrained_model: None # or path to a pretrained model to load
        using_bn: true # apply batch normalization in the model
        det_thresh: 0.001 # point confidence threshold, default, 1/65
        nms: 4 # nms window size
        topk: -1 # keep top-k points, -1, keep all
     ...
    data:
        name: coco #synthetic
        image_train_path: ['./data/mp_coco_v2/images/train2017',] #several data sets can be list here
        label_train_path: ['./data/mp_coco_v2/labels/train2017/',]
        image_test_path: './data/mp_coco_v2/images/test2017/'
        label_test_path: './data/mp_coco_v2/labels/test2017/'
        ...
        data_dir: './data/hpatches' #path to the hpatches dataset
        export_dir: './data/repeatibility/hpatches/sp' #dir where to save the output data
    solver:
        model_name: sp #Prefix of the model name you want
    ```

