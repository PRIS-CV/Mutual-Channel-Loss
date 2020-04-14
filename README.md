# The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification

Code release for The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification (TIP 2020)
[DOI](https://doi.org/10.1109/TIP.2020.2973812 "DOI")


## Changelog
2020/04/13 Upload the pre-trained ResNet50 model and output information.

## Dataset
### CUB-200-2011

## Requirements

- python 3.6
- PyTorch 1.2.0
- torchvision

## Training

- Download datasets
- Train: `python CUB-200-2011.py`, the alpha is the hyper-parameters of the  `MC-Loss`
- Description : PyTorch CUB-200-2011 Training with VGG16 (TRAINED FROM SCRATCH).

- Train: `python MC_ResNet50.py`, the alpha is the hyper-parameters of the  `MC-Loss`
- Description : 1) PyTorch CUB-200-2011 Training with ResNet50 (USING THE PRE-TRAINED MODEL). 2） MC_ResNet50.out is the output information.

## Citation
If you find this paper useful in your research, please consider citing:
```
@ARTICLE{9005389, 
author={D. {Chang} and Y. {Ding} and J. {Xie} and A. K. {Bhunia} and X. {Li} and Z. {Ma} and M. {Wu} and J. {Guo} and Y. {Song}}, 
journal={IEEE Transactions on Image Processing}, 
title={The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification}, year={2020}, volume={29}, number={}, 
year={2020},
pages={4683-4695}, 
keywords={Feature extraction;Training;Visualization;Automobiles;Task analysis;Data mining;Manuals;Fine-grained image classification;deep learning;loss function;mutual channel}, 
doi={10.1109/TIP.2020.2973812}, 
ISSN={1941-0042}, 
month={},} 
```


## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- changdongliang@bupt.edu.cn
- mazhanyu@bupt.edu.cn
