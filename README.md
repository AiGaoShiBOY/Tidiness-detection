# Tidiness Detction Network
This is a simple network for tidiness detection.
# How to start?
* You need to install
> * Pytorch
> * PIL
* modify my_route to your own dataset route in `Dataset.py` and `DatasetFor5layer.py`
* run `main.py`.
# Available functions
* 5-layer-CNN 
* Resnet18
* FC-Finetune Resnet18

When you run `main.py`, you can see :
> 1--------2layer-Net<br>
  2--------Resnet18<br>
  3--------VGG16<br>
  4--------FC-Finetune Pretrained Resnet18

Input corresponding number to choose Net.

# Update History
* 2020.3.27 A 5-layer CNN. Val-Acc:0.65
* 2020.3.30 FC-Finetune Pretrained Resnet18: Val-Acc:0.9
* 2020.3.31 integration
* 2020.3.31 Resnet18
* 2020.3.31 5-layer CNN refinement
* 2020.3.31 Visualization/Statistics/Training TIME documentary

# GPU Available
Automatically adapted GPU for training.