# Heuristic Dropout: An Efficient Regularization Method for Medical Image Segmentation Models

Official implementation of [Heuristic Dropout: An Efficient Regularization Method for Medical Image Segmentation Models](https://ieeexplore.ieee.org/abstract/document/9747409). 

### Usage

We provide an out-of-the-box [implementation](./HeuristicDropout.py) of the proposed Heuristic Dropout. To use the Heuristic Dropout, simply replace the original `nn.Dropout` in your medical image segmentation models with it.

### Cite
If you find this work useful, please consider citing the corresponding paper:
<pre/>
@inproceedings{shi2022heuristic,
  title={Heuristic Dropout: An Efficient Regularization Method for Medical Image Segmentation Models},
  author={Shi, Dachuan and Liu, Ruiyang and Tao, Linmi and Yuan, Chun},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1101--1105},
  year={2022},
  organization={IEEE}
}
</pre>