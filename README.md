## [ Salient Object Detection](http://)

code for paper, “CRRNet:Channel Relation Reasoning network for Salient Object Detection”
by Shuyong Gao, Wei Zhang, Hong Lu, Yan Wang, Qianyu Guo, Wenqiang Zhang

### Introduction
![pipline](https://user-images.githubusercontent.com/34783695/128590805-6f471579-9d31-4cbc-9647-e0326b37896d.png)

### Prerequisites
- Python 3.6
- Pytorch 1.3
- OpenCV 4.0

### Clone repository
```
git clone https://
cd CRRNet
```

### Download dataset
Download the following datasets and unzip them
- PASCAL-S
- ECSSD
- HKU-IS
- DUT-OMRON
- DUTS
- THUR15K

### Training & Evaluation
- If you want to train the model by yourself, please download the [pretrained model]() into res folder
- Split the ground truth into body map and detail map, which will be saved into ```data/DUTS/body-origin``` and ```data/DUTS/detail-origin```
```
python utils.py
```
- Train the model and get the predicted body and detail maps, which will be saved into data/DUTS/body and data/DUTS/detail

### Testing & Evaluate


### Saliency maps & Trained model

- saliency maps: [google](http://)
- trained model: [google](http://)

![image](https://user-images.githubusercontent.com/34783695/128591389-c2a9fb3c-78d1-4f4c-a84b-1149ff125850.png)
![image](https://user-images.githubusercontent.com/34783695/128591404-ddd55757-76b4-46a9-b80a-3a77e0e44bc4.png)

### Citation
```

```
### Thanks
```
@inproceedings{ldf,
  author    = {J. {Wei} and S. {Wang} and Z. {Wu} and C. {Su} and Q. {Huang} and Q. {Tian}},
  title     = {F{\({^3}\)}Net: Fusion, Feedback and Focus for Salient Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {13022-13031},
  year      = {2020},
}

@inproceedings{f3net,
  author    = {Jun Wei and Shuhui Wang and Qingming Huang},
  title     = {F{\({^3}\)}Net: Fusion, Feedback and Focus for Salient Object Detection},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages     = {12321--12328},
  year      = {2020},
}
```



