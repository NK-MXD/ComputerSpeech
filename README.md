# 仓库说明和实验说明

> 👋 这里是我的语音处理技术仓库, 存放了一些课程常用代码和作业代码

## 实验1: fbank特征提取

给定一段语音信号（16KHZ Wav PCM），提取80维Log Mel Spectrogram（Fbank）特征，并画图。

额外工作: 
+ 抽取spectrogram特征, 并可视化
+ 抽取MFCC特征，并可视化
+ 抽取PLP特征，并可视化

## 实验2: dtw距离计算

理解DTW算法原理，参考博客如下

1. https://blog.csdn.net/chenxy_bwave/article/details/121052541
2. http://alexminnaar.com/2014/04/16/Time-Series-Classification-and-Clustering-with-Python.html

TODO1：理解DTW算法，写入递推公式

TODO2：计算yes1和no2的dtw距离和yes3和yes1的dtw距离

TODO3：将yes1和yes3两个音频帧与帧之间的对应关系画出来

提交
1. TODO2 之中打印的距离
2. TODO3 之中输出的图表

