# python-tutorial

[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/shibing624/python-tutorial.svg)](https://github.com/shibing624/python-tutorial/graphs/contributors)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_vesion](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/python-tutorial.svg)](https://github.com/shibing624/python-tutorial/issues)
[![Wechat Group](http://vlog.sfyc.ltd/wechat_everyday/wxgroup_logo.png?imageView2/0/w/60/h/20)](#Contact)


Python教程，包括：Python基础，Python高级特性，面向对象编程，多线程，Web开发，数据库，数据科学，NLP，CV，深度学习库使用教程。



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
| [01_base](01_base)       | Python基础    | 提供了数据类型、字符串、list、条件判断、循环、函数、文件、多进程的使用例子。 |
| [02_advanced](02_advanced)       | Python高级特性    | 提供了数据库、高阶函数、迭代器、面向对象编程的使用例子。 |
| [03_data_science](03_data_science)    | 数据科学 | 提供了常用数据科学库（numpy、scipy、scikit-learn、pandas）的使用例子。 |
| [04_flask](04_flask)      | Flask开发    | 提供了Web框架Flask的使用例子。 |
| [05_spider](05_spider) | 爬虫    | 提供了爬虫的实现例子。|
| [06_deep_learning](06_deep_learning) | 深度学习库    | 提供了常用深度模型库（TensorFlow、Keras、PyTorch）的使用例子。|
| [08_nlp](08_nlp)       | 自然语言处理任务    | 提供了NLP任务的模型使用的例子。 |
| [10_cv](10_cv) | 计算机视觉任务    | 提供了CV任务的使用例子。|
| [11_speech](11_speech) | 语音识别任务    | 提供了语音识别任务的使用例子。|
| [13_tool](13_tool) | 实用工具    | 提供了常用的实用工具，包括文件解析、微信机器人、统计脚本等例子。|



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


# Get Started

教程代码大多数为Jupyter Notebook书写（文件后缀.ipynb），如下所示：
![notebook](./docs/imgs/readme_img.png)

使用Jupyter Notebook打开学习：
1. 下载Python：建议使用Anaconda，Python环境和包一键装好，[Python3.7 版本](https://www.anaconda.com/products/individual)
2. 下载本项目：可以使用`git clone`，或者下载zip文件，解压到电脑
3. 打开Jupyter Notebook：打开终端，`cd`到本项目所在的文件夹，执行：```jupyter notebook ```，浏览器打开`01_base/01_字符串类型.ipynb`，跟随介绍交互使用

# Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/python-tutorial.svg)](https://github.com/shibing624/python-tutorial/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：
加我*微信号：xuming624, 备注：个人名称-NLP* 进Python-NLP交流群。

<img src="docs/we_image.jpeg" width="200" /><img src="docs/wechat.jpeg" width="200" />

读后有收获可以支付宝打赏作者喝咖啡，读后有疑问请加微信群讨论：

<img src="docs/wechat_zhifu.png" width="150" />

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
3. [Python4DataScience.CH](https://github.com/catalystfrank/Python4DataScience.CH)
4. [Python-100-Days](https://github.com/jackfrued/Python-100-Days)
5. [flask-tutorial](https://github.com/greyli/flask-tutorial)