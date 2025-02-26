# AutoProg

#### Modified from [VOLO](https://github.com/sail-sg/volo) and  [Token Labeling](https://github.com/zihangJiang/TokenLabeling), not the final version.

This project contains code of our paper [***Automated Progressive Learning for Efficient Training of Vision Transformers***](https://arxiv.org/pdf/2203.14509) (CVPR 2022).

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/61453811/160517880-c6349005-a2ab-4587-a51f-644cf4a7360e.png" width=95%/></p>
<p align="center">
AutoProg achieves efficient training by automatically increasing the training overload on-the-fly.</p>

<p align="center">
<img alt="image" src="https://user-images.githubusercontent.com/61453811/160518329-1f4798f8-e423-4f3a-add4-9369c3fcc5c0.png" width=95%/></p>
<p align="center">
AutoProg can accelerate ViTs training by up to 85.1% with no performance drop.
</p>

#### *[New!]* Check out the extension of this work on diffusion models! ***AutoProg-Zero*** [ [arXiv](https://arxiv.org/abs/2410.00350) | [code](https://github.com/changlin31/AutoProg-Zero) ]. 

## 1. Requirements

- torch>=1.7.0 (1.8.0 is tested and recommended); torchvision>=0.8.0; [timm](https://github.com/rwightman/pytorch-image-models)==0.4.5; [tlt](https://github.com/zihangJiang/TokenLabeling)==0.1.0; pyyaml; apex-amp

- Docker is recommended as this repo requires some outdated packages. All the requirements will be automatically installed via `Dockerfile`. Install Docker, then run:
```
Docker build Dockerfile
```

- data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```


## 2. Train

- As we use token labeling, please download the token labeling data in [Google Drive](https://drive.google.com/file/d/1Cat8HQPSRVJFPnBLlfzVE0Exe65a_4zh/view) or [BaiDu Yun](https://pan.baidu.com/s/1YBqiNN9dAzhEXtPl61bZJw) (password: y6j2), details about token labling are in [here](https://github.com/zihangJiang/TokenLabeling).

- Train volo_d1 with AutoProg for 100 epochs: first modify the data paths in scripts/train_auroprog.sh, then run:
```bash
sh scripts/train_autoprog.sh
```

## Citation
If you use our code for your paper, please cite:
```bibtex
@inproceedings{li2022autoprog,
  author = {Li, Changlin and 
            Zhuang, Bohan and 
            Wang, Guangrun and
            Liang, Xiaodan and
            Chang, Xiaojun and
            Yang, Yi},
  title = {Automated Progressive Learning for Efficient Training of Vision Transformers},
  booktitle = {CVPR},
  year = 2022,
}
```

## LICENSE

This repo is under the Apache-2.0 license.
