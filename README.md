# CaDRe-Net: Coupled Calibration and Disruption-Repair for Fine-Grained Few-Shot Image Classification
## Data Preparation
The following datasets are used in our paper:

Stanford Dogs: [Dataset Page](http://vision.stanford.edu/aditya86/ImageNetDogs/)

Stanford Cars:[Dataset Page](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view)

CUB_200_2011: [Dataset Page](https://www.vision.caltech.edu/datasets/cub_200_2011/)

## Requirements
```
python=3.12+
PyTorch=2.8+
torchvision=0.4.2
pillow=6.2.1
numpy=1.18.1
h5py=1.10.2
chardet==3.0.4
cloudpickle==1.1.1
colorama==0.4.1
```

🚀 Usage

```
python train.py --dataset CUB  --model Conv4 --epoch 150 --n_shot 5
```
All code will be made public after the paper is accepted
