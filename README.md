# python-tutorial
python教程，包括：python基础、numpy、scipy、python进阶、matplotlib、OOP、tensorflow、keras、pandas、NLP analysis.



在本Python教程包含了一些范例，涵盖了大多数常见Python日常脚本任务，是入门Python的学习资料，也可以作为工作中上手Python的参考实现。


## nlp-tutorial的例子清单

| **目录**  | **主题**                                           | 简要说明                                                      |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 01_base       | Python基础    | 提供了数据类型、字符串、list、条件判断、循环的使用例子。 |
| 02_advanced       | Python高级特性    | 提供了切片、迭代、生成器、迭代器的使用例子。 |
| 03_oop       | 面向对象编程    | 提供了多重继承、定制类、枚举类、设计模式的使用例子。   |
| 04_thread       | 多线程    | 提供了多线程、多进程的例子。 |
| 05_web      | Web开发    | 提供了Web框架、模板、Web API的使用例子。 |
| 06_database       | 数据库    | 提供数据库（包括SQLite、MySQL、SQLAlchemy）使用例子。 |
| 07_statistics    | 科学计算 | 提供了几个常用科学计算库（numpy、scipy、scikit-learn、pandas）的使用例子。 |
| 08_nlp       | 自然语言处理任务    | 提供了NLP任务的模型使用的例子。 |
| 09_deep_learning | 深度学习库    | 提供了常用深度模型库（TensorFlow、PyTorch）的使用例子。|
| 10_cv | 计算机视觉任务    | 提供了CV任务的使用例子。|
| 11_speech | 语音识别任务    | 提供了语音识别任务的使用例子。|
| 12_tool | 文件解析工具    | 提供了常用的实用工具，包括文件解析、微信机器人、统计脚本等例子。|



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