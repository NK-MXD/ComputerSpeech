# 实验三实验进度与问题记录

## 实验内容

+ Corpus: https://github.com/neubig/anlp-code/tree/main/data/ptb-text
+ Reference code: https://github.com/neubig/anlp-code/blob/main/03-lm/nn-lm.py
+ Result: 
    1. Your code
    2. Visualization of loss on train-set and dev-set per epoch (tensorboard)
        it is a plus if you can improve the mode
        It is a plus if you can calculate the perplexity of “Jane went to the store” and “store to Jane went the” using the trained model


## 实验进度

- [x] 复现FFNNLM，提交输出结果截图：50分
- [x] 回答雨课堂中两个问题：10分
- [x] 使用tensorboard可视化loss曲线，提交结果截图：30分
- [ ] 计算给定两个句子的复杂度，提交对应代码和结果截图：10分
- [ ] 改进模型（附加，10分）


## 实验内容

### visualize the curve

```
1> install pytorch
2> install tensorboard
3> write code to generate data which you want to visualize
4> run “tensorboard –logdir <your log dir>
5> run browser to view your curve
```

