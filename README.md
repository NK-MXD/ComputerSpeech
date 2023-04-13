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

## 实验3: FFNNLN

1. 根据提供的语料库和参考代码，构建FFNN语言模型，提交运行结果截图。使用tensorboard可视化模型在训练集和验证集上的loss曲线，以每个epoch为单位。
2. 使用训练好的语言模型，计算以下两句话的困惑度：“Jane went to the store”和“store to Jane went the”改进模型

> 参考资料: 
> + 论文  https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
> + 博客  https://blog.csdn.net/blmoistawinde/article/details/104966127

