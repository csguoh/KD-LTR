

# KD-STR

An official Pytorch implement of the paper "One-stage Low-resolution Text Recognition with High-resolution Knowledge Transfer" (MM2023).

Authors: *Hang Guo, Tao Dai, Mingyan Zhu, GuangHao Meng, Bin Chen, Zhi Wang, Shu-Tao Xia*.

## Motivation

This work focus on the problem of text recognition on the low-resolution. A novel knowledge distillation framework is proposed, which can directly adapt the text recognizer to low-resolution. We hope that our work can inspire more studies on one-stage low-resolution text recognition.

![image-20230730095434441](C:\Users\郭航\AppData\Roaming\Typora\typora-user-images\image-20230730095434441.png)


## Pipeline
The architecture of the proposed framework is as follows.



![image-20230730095345453](C:\Users\郭航\AppData\Roaming\Typora\typora-user-images\image-20230730095345453.png)





## Pre-trained Weight

We refer to the student model adapted to low-resolution inputs as ABINet-LTR, MATRN-LTR and PARSeq-LTR, respectively. As pointed out in the paper, since the input images between the two branches are of different resolutions, we modified the convolution stride (for CNN backbone) or patch sizes (for ViT backbone) to ensure the consistency of the deep visual features. The pretrained weights can be downloaded as follows.

|    Model    | [ABINet-LTR](https://drive.google.com/file/d/1DihgbIyMwNMD2N1dVzkfWreMbXug0hiS/view?usp=drive_link) | [MATRN-LTR](https://drive.google.com/file/d/1-cZJmL6UwqgF1RJF7AOh26oVM1RQbcVF/view?usp=drive_link) | [PARSeq-LTR](https://drive.google.com/file/d/1acjJ9uT4UAFihPHwyrbRIqnTAsccr1VK/view?usp=drive_link) |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Performance |                            72.45%                            |                            73.27%                            |                            78.23%                            |

Please be noted that the pre-trained HR teacher model is still needed for both training and testing, you can download the model in their coresponding offical github repository, i.e.  [ABINet](https://github.com/FangShancheng/ABINet), [MATRN](https://github.com/byeonghu-na/MATRN) and [PARSeq](https://github.com/baudm/parseq). 

## Datasets

In this work, we use STISR datasets TextZoom and five STR benchmarks, i.e.,  ICDAR2013, ICDAR2015, CUTE80, SVT and SVTP for model comparison. All the datasets are in `lmdb` format.  One can download these datasets from the following table.

|   Datasets    |                           TextZoom                           |                             IC13                             |                             IC15                             |                            CUTE80                            |                             SVT                              |                             SVTP                             |
| :-----------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Download Link | [link](https://drive.google.com/drive/folders/1fdnW0DCzSBXF2_p4zo1JIxV8kxplCdM0?usp=drive_link) | [link](https://drive.google.com/drive/folders/1-cVllciQs7f56lLSf5A8_9MRfIDLfUDy?usp=drive_link) | [link](https://drive.google.com/drive/folders/1fdnW0DCzSBXF2_p4zo1JIxV8kxplCdM0?usp=drive_link) | [link](https://drive.google.com/drive/folders/1fwAWVGftPgdQG5CJfU-X79_BhgfGL3pt?usp=drive_link) | [link](https://drive.google.com/drive/folders/11iR4e3mFCy40fRCFAWbgHTPBm7KuH0yY?usp=drive_link) | [link](https://drive.google.com/drive/folders/1mjqP9vWAs5u1Ob-n3ye2grGouYzmtNqf?usp=drive_link) |

## How to Run?

We have set some default hype-parameters in the `config.yaml` and `main.py`, so you can directly implement training and testing after you modify the path of datasets and pre-trained model.  

### Training

```
python main.py
```

### Testing

```
python main.py --go_test
```



## Main Results

### Quantitative Comparison

![image-20230730095511775](C:\Users\郭航\AppData\Roaming\Typora\typora-user-images\image-20230730095511775.png)

### Qualitative Comparison

![image-20230730095526767](C:\Users\郭航\AppData\Roaming\Typora\typora-user-images\image-20230730095526767.png)

### Robustness Comparison

![image-20230730095547636](C:\Users\郭航\AppData\Roaming\Typora\typora-user-images\image-20230730095547636.png)

## Citation

If you find our work helpful, please consider citing us.

```

```
