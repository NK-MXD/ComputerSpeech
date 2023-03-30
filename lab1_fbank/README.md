# 实验1进度与问题记录

## 实验描述
1. 给定一段语音信号（16KHZ Wav PCM），提取80维Log Mel Spectrogram（Fbank）特征，并画图。
2. 选做实验内容: 抽取spectrogram特征, 并可视化~~已完成~~, 抽取MFCC特征，并可视化~~已完成~~, 抽取PLP特征，并可视化~~已完成~~。
## 实验过程
实验过程基本按照下图所示的实验步骤进行实验   

![202303090804169|500](https://image-1305894911.cos.ap-beijing.myqcloud.com/Obsidian/202303090805998.png)
### Fbank特征抽取

### 实验代码

```python
# step 1: 读入语音
sample_rate, signal = wavfile.read('./speech_feature/wav/我爱南开.wav')
# step 2: 预加重
sig = pre_emphasis(signal)
# step 3: 分帧
frame_sig = framing(sig, sample_rate)
# step 4: 加窗
frame_sig = add_window(frame_sig, sample_rate)
# step 5: 快速傅里叶变换
mag_frames = my_fft(frame_sig)
# step 6: 幅值平方, 变为功率谱
frame_pow = stft(frame_sig)
# step 7: mel滤波器: 注意此时mel滤波器要设置为80维
filter_banks = mel_filter(frame_pow, sample_rate, n_filter=80)
# step 8: 对数功率
filter_banks = log_pow(filter_banks)
# 绘图
plot_spectrogram(filter_banks.T, "Filter Banks", "mel图")
```
### 实验结果

![output.png|500](https://image-1305894911.cos.ap-beijing.myqcloud.com/Obsidian/202303152238042.png)
### 其它特征抽取
### 实验代码
以下代码是抽取spectrogram特征, 并可视化

```python
# 直接对数功率, 不进行mel滤波
spec = log_pow(frame_pow)
# 绘图
plot_spectrogram(spec, "spectrogram", "spectiogram可视化")
```

以下代码是抽取MFCC特征, 并可视化, 其中离散余弦变换在这里类似逆傅里叶变换的作用。

```python
# 离散余弦变换
mfcc = discrete_cosine_transform(filter_banks)
# 绘图
plot_spectrogram(mfcc, "mfcc", "mfcc可视化")
```

以下代码是抽取PLP特征, 并可视化, 在PLP特征的抽取中借鉴了`Spafe`库函数中的代码, 使用了Spafe库函数中的API来帮助我们实现PLP特征抽取

```python
from spafe.features.lpc import __lpc_helper, lpc2lpcc
from spafe.utils.vis import show_features
# a. 获取一个berk滤波器 bark_filter_banks
fbanks = bark_filter_banks()

# b. 进行bark滤波
auditory_spectrum = np.dot(a=frame_pow, b=fbanks.T)

# c. 等响度预加重
E = lambda w: ((w**2 + 56.8 * 10**6) * w**4) / (
    (w**2 + 6.3 * 10**6)
    * (w**2 + 0.38 * 10**9)
    * (w**6 + 9.58 * 10**26)
)
Y = [E(w) for w in auditory_spectrum]

# d. 强度响度转换
L = np.abs(Y) ** (1 / 3)

# e. ifft 逆傅里叶变换
inverse_fourrier_transform = np.absolute(np.fft.ifft(L, 512))

# f. compute lpcs and lpccs 线性预测（lpc)
lpcs = np.zeros((L.shape[0], 13))
lpccs = np.zeros((L.shape[0], 13))
for i in range(L.shape[0]):
    a, e = __lpc_helper(inverse_fourrier_transform[i, :], 13 - 1)
    lpcs[i, :] = a
    lpcc_coeffs = lpc2lpcc(a, e, 13)
    lpccs[i, :] = np.array(lpcc_coeffs)

# 绘图
show_features(lpccs, "Perceptual linear predictions", "PLP Index", "Frame Index")
```

在PLP特征的抽取过程中, 我们也可以使用对应的库函数直接生成PLP特征

```python
# spafe库plp抽取
from scipy.io.wavfile import read
from spafe.features.rplp import plp
from spafe.utils.preprocessing import SlidingWindow
from spafe.utils.vis import show_features

plps = plp(signal)

# visualize features
show_features(plps, "Perceptual linear predictions", "PLP Index", "Frame Index")
```

另外, 我们也可以借助`spafe`库函数实现CQCC特征提取

```python
from spafe.features.cqcc import cqt_spectrogram
from spafe.utils.vis import show_spectrogram
from spafe.utils.preprocessing import SlidingWindow
from scipy.io.wavfile import read

# read audio
fpath = "speech_feature\wav\我爱南开.wav"
fs, sig = read(fpath)

# compute spectrogram
qSpec = cqt_spectrogram(sig,
                        fs=fs,
                        pre_emph=0,
                        pre_emph_coeff=0.97,
                        window=SlidingWindow(0.03, 0.015, "hamming"),
                        nfft=2048,
                        low_freq=0,
                        high_freq=fs/2)

# visualize spectrogram
show_spectrogram(qSpec,
                 fs=fs,
                 xmin=0,
                 xmax=len(sig)/fs,
                 ymin=0,
                 ymax=(fs/2)/1000,
                 dbf=80.0,
                 xlabel="Time (s)",
                 ylabel="Frequency (kHz)",
                 title="CQT spectrogram (dB)",
                 cmap="jet")
```
### 实验结果
spectrogram特征抽取可视化结果: 

![202303152252921|500](https://image-1305894911.cos.ap-beijing.myqcloud.com/Obsidian/202303152252921.png)

MFCC特征抽取可视化结果: 

![202303152253105|500](https://image-1305894911.cos.ap-beijing.myqcloud.com/Obsidian/202303152253105.png)

PLP特征抽取可视化结果: 

![202303152248337|500](https://image-1305894911.cos.ap-beijing.myqcloud.com/Obsidian/202303152248337.png)

## 参考资料
全部实验代码已上传 [我的计算机语音处理技术github仓库](https://github.com/NK-MXD/ComputerSpeech/tree/main)

> [!ob-example] 参考资料
> + 博文 [(72条消息) Python语音信号特征-感知线性预测系数PLP_虾米的圈的博客-CSDN博客](https://blog.csdn.net/weixin_42485817/article/details/107590846)
> + 博文 [(72条消息) 语音识别-特征提取(Python实现）WuJia_的博客-CSDN博客](https://blog.csdn.net/WuJia_/article/details/107044859)
> + 文档 [📄 API documentation — 🧠 SuperKogito/Spafe 0.3.2 documentation](https://superkogito.github.io/spafe/api_documentation.html)


