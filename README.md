# NorCal in Detectron2
[**On Model Calibration for Long-Tailed Object Detection and Instance Segmentation**](https://arxiv.org/abs/2107.02170). In Neural Information Processing Systems (NeurIPS), 2021.
[Tai-Yu Pan*](https://scholar.google.com/citations?user=M1_TnJsAAAAJ&hl=en&authuser=5), [Cheng Zhang*](https://czhang0528.github.io/), 
[Yandong Li](https://cold-winter.github.io/), [Hexiang Hu](http://www.hexianghu.com/), [Dong Xuan](https://web.cse.ohio-state.edu/~xuan.3/), 
[Soravit Changpinyo](http://www-scf.usc.edu/~schangpi/), [Boqing Gong](http://boqinggong.info/), [Wei-Lun Chao](https://sites.google.com/view/wei-lun-harry-chao). 

## Introduction
Vanilla models for object detection and instance segmentation suffer from the heavy bias toward detecting frequent objects in the long-tailed setting. Existing methods address this issue mostly during training, e.g., by re-sampling or re-weighting. 

In this paper, we investigate a largely overlooked approach -- post-processing calibration of confidence scores. We propose NorCal, Normalized Calibration for long-tailed object detection and instance segmentation, a simple and straightforward recipe that reweighs the predicted scores of each class by its training sample size. We show that separately handling the background class and normalizing the scores over classes for each proposal are keys to achieving superior performance. On the LVIS dataset, NorCal can effectively improve nearly all the baseline models not only on rare classes but also on common and frequent classes. Finally, we conduct extensive analysis and ablation studies to offer insights into various modeling choices and mechanisms of our approach.

## Installation
Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Citation
Please cite with the following bibtex if you find it useful.
```
@inproceedings{pan2021norcal,
  title={On Model Calibration for Long-Tailed Object Detection and Instance Segmentation},
  author={Pan, Tai-Yu and Zhang, Cheng and Li, Yandong and Hu, Hexiang and Xuan, Dong and Changpinyo, Soravit and Gong, Boqing and Chao, Wei-Lun},
  booktitle = {NeurIPS},
  year={2021}
}
```