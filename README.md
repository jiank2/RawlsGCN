# RawlsGCN: Towards Rawlsian Difference Principle on Graph Convolutional Network
Implementations of 'RawlsGCN: Towards Rawlsian Difference Principle on Graph Convolutional Network', WWW'22

## Requirements
Main dependency: pytorch

Tested under python 3.8, pytorch 1.9

## Run
RawlsGCN-Graph:
```
python train.py --model rawlsgcn_graph
```

RawlsGCN-Grad:
```
python train.py --model rawlsgcn_grad
```

## Citation
If you find this repository useful, please kindly cite the following paper:

```
@inproceedings{kang2022rawlsgcn,
  title={Rawlsgcn: Towards rawlsian difference principle on graph convolutional network},
  author={Kang, Jian and Zhu, Yan and Xia, Yinglong and Luo, Jiebo and Tong, Hanghang},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={1214--1225},
  year={2022}
}
```