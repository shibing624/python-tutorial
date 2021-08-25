# python-tutorial

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/python-tutorial.svg)](https://github.com/shibing624/python-tutorial/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_vesion](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/python-tutorial.svg)](https://github.com/shibing624/python-tutorial/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)


python教程，包括：python基础、numpy、scipy、python进阶、matplotlib、OOP、tensorflow、keras、pandas、NLP analysis.



在本Python教程包含了一些范例，涵盖了大多数常见Python日常脚本任务，是入门Python的学习资料，也可以作为工作中编写Python脚本的参考实现。
以下所有实现均在python3环境下测试。


**Guide**

- [Tutorial](#python-tutorial的例子清单)
- [Get Started](#get-started)
- [Contact](#Contact)
- [Cite](#Cite)
- [Reference](#reference)


# python-tutorial的例子清单

| **目录**  | **主题**            | 简要说明                              |
| --------------------- | -------------------------------------------- | ---------------------------- |
| [01_base](01_base)       | Python基础    | 提供了数据类型、字符串、list、条件判断、循环的使用例子。 |
| 02_advanced       | Python高级特性    | 提供了切片、迭代、生成器、迭代器的使用例子。 |
| 03_oop       | 面向对象编程    | 提供了多重继承、定制类、枚举类、设计模式的使用例子。   |
| 04_thread       | 多线程    | 提供了多线程、多进程的例子。 |
| 05_web      | Web开发    | 提供了Web框架、模板、Web API的使用例子。 |
| 06_database       | 数据库    | 提供数据库（包括SQLite、MySQL、SQLAlchemy）使用例子。 |
| 07_data_science    | 数据科学 | 提供了几个常用数据科学库（numpy、scipy、scikit-learn、pandas）的使用例子。 |
| 08_nlp       | 自然语言处理任务    | 提供了NLP任务的模型使用的例子。 |
| 09_deep_learning | 深度学习库    | 提供了常用深度模型库（TensorFlow、PyTorch）的使用例子。|
| 10_cv | 计算机视觉任务    | 提供了CV任务的使用例子。|
| 11_speech | 语音识别任务    | 提供了语音识别任务的使用例子。|
| 12_spider | 爬虫    | 提供了爬虫的实现例子。|
| 13_tool | 实用工具    | 提供了常用的实用工具，包括文件解析、微信机器人、统计脚本等例子。|



### 09_deep_learning
#### keras 
> * bAbi: 阅读理解任务
  * 记忆网络（memory network）实现阅读理解任务
  * RNN网络实现
> * keras 应用
  * 01.base: 认识keras变量
  * 02mlp_multi_classification: 多层感知器多分类深度网络
  * 03mlp_binary_classification: 多层感知器二分类深度网络
  * 04vgg_conv: VGG，图像经典卷积网络结构
  * 05lstm_classification: LSTM分类网络
  * 06sequential: 序列模型，模型保存
  * 07shared_lstm: 模型参数共享
  * 08imdb_fasttext: fasttext网络结构实现二分类
  * 09fasttext_multi_classification: fasttext的深度网络结构实现多分类
  * 10seq2seq: 法语到英语的翻译任务，seq2seq模型
  * 11lstm_text_generation: LSTM模型实现文本生成任务
  * 12rnn_num_add: RNN网络学习三位数以内的加法运算
  * 13rnn_num_multiplication: RNN网络学习三位数以内的乘法运算

#### xgboost 
文本特征分类

[lr.py](lr.py)
lr 文本分类

[xgb.py](xgb.py)
xgboost 文本分类


[xgb_lr.py](xgb_lr.py)
xgboost 提取特征之间的关系，再用lr文本分类

# Get Started


![notebook](./docs/imgs/readme_img.png)
<img src="docs/imgs/readme_img.png" width="500" />



# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/python-tutorial.svg)](https://github.com/shibing624/python-tutorial/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：个人名称-NLP* 进Python-NLP交流群。

<img src="docs/wechat.jpeg" width="200" />


# Cite

如果你在研究中使用了python-tutorial，请按如下格式引用：

```latex
@software{python-tutorial,
  author = {Xu Ming},
  title = {python-tutorial: Python3 Tutorial for Beginners},
  year = {2021},
  url = {https://github.com/shibing624/python-tutorial},
}
```

# License


授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加python-tutorial的链接和授权协议。


# Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

 - 在本地进行单元测试
 - 确保所有单测都是通过的

之后即可提交PR。

# Reference

1. [缪雪峰Python3教程](https://www.liaoxuefeng.com/wiki/1016959663602400)
2. [PythonDataScienceHandbook](https://github.com/jakevdp/PythonDataScienceHandbook)
