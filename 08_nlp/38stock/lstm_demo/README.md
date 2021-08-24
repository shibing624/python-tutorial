# hs300_stock_predict
该项目用于对hs300股票的预测，包括股票下载，数据清洗，LSTM 模型的训练，测试，以及实时预测。<br>

## 文件构成
    1、data_utils.py 用于股票数据下载，清洗，合并等。该文件共有9个函数。
get_stock_data(code, date1, date2, filename, length=-1)<br>
该函数用于下载股票数据，保存开、高、收、低、量、涨跌幅等6维数据。<br>
由于用的tushare接口，因此只能下载最近两年的数据。（从新浪网易财经的数据爬虫接口后续开放）<br>
共有`5个`参数<br>
`code`是需要下载的股票代码，如000001是平安银行的股票代码，输入'000001'既下载平安银行的股票数据。<br>
`date1`是开始日期，格式如"2019-01-03",`date2`是结束日期，格式同上。<br>
`filename`是存放数据的目录，如"D:\data\"<br>
`length`是筛选股票长度，默认为-1，既对下载的股票数据长度上不做筛选，如果人为指定长度如200，既会将时间长度200以下的数据剔除不予以保存。<br><br>
get_hs300_data(date1, date2, filename)<br>
该函数用于下载沪深300指数数据，参数格式同get_stock_data<br><br>
update_stock_data(filename)<br>
该函数将股票数据从本地文件的最后日期更新至当日，`filename`是指定的单只股票路径名称，如"d:\data\000001.csv"<br><br>
get_data_len(file_path)<br>
该函数过去单只股票的时间长度，`file_path`是单只股票路径名称，如"d:\data\000001.csv"<br><br>
select_stock_data(file1, file2, date1, date2)<br>
该函数对已经再本地的文件按照日期筛选，`date1`是开始数据，`date2`是结束数据，`file1`是源文件夹，`file2`是筛选日期后文件存放的文件夹<br><br>
crop_stock(df, date)<br>
该函数暂时不使用<br><br>
fill_stock_data(target, sfile, tfile)<br>
该函数按照沪深300指数的时间长度来对个股停盘数据进行填充，填充为该股上一交易日的数据。该函数是对所选文件夹下所有文件进行处理。<br>
注意，如果开始日期是属于停牌的，那么该段停牌将不会被填充，后续会有更新。<br>
`target`为参照股票，一般选择同时间段的沪深300指数文件，`sfile`为原文件夹，`tfile`为填充完要存放文件夹。<br><br>
merge_stock_data(filename, tfile)<br>
该函数将多个文件按序合并为一个文件，如讲沪深300只个股文件合并为一个总文件，方便后续模型输入。<br>
`filename`是需要合并的文件夹路径，`tfile`是存放合并后文件的文件夹路径。<br><br>
quchong(file)<br>
该函数暂不使用。<br><br>

    2、dataprocess.py 用于训练数据的处理，归一化等，模型的输入都由该文件的接口输出提供。
get_train_data(batch_size=args.batch_size, time_step=args.time_step)<br>
该函数用于处理训练数据，参数默认，有配置文件给定。该函数返回五个变量：`batch_index`训练集分批处理的批次, `val_index`验证集批次, `train_x_1`, 训练集输入，`train_y_1`, 训练集标签，`val_x`, y验证集输入，`val_y`验证集标签<br>
备注：由于整个数据处理是对多只股票合成的总文件处理，所以在时间步长迭代添加时须在各自股票时间长度内完成，因此，需要在配置文件中指定股票长度。<br><br>
get_test_data(time_step=args.time_step)<br>
该函数用于处理测试集数据，返回两个变量：`test_x`测试集输入, `test_y`测试集标签。<br><br>
get_update_data(time_step=args.time_step)<br>
该函数将更新数据加历史数据的前time_step-1拼接,用于整批处理，如2019 1-3月数据和2018.12的数据拼接，返回拼接后的`train_x`, `train_y`<br>
get_predict_data(file_name)<br>
该函数完成下载实时股票数据，与之前的数据拼接后返回输入x。有一个参数需要填充，`file_name`既要预测的单只股票文件。

    3、config.py 配置文件，所有接口内超参数，路径等，在该文件修改，即可在全局生效。

    4、lstm_model 模型，包括训练，微调，测试，及预测。
train_lstm(time_step=args.time_step, val=True)<br>
用于训练的函数，val既是否验证，默认开启。其数据来自`get_train_data()`<br><br>
fining_tune_train(time_step=args.time_step)<br>
用于微调模型，如新增数据在原模型继续训练，或者迁移学习等。其数据可以来自`get_update_data()`<br><br>
test(time_step=args.time_step)<br>
用于测量测试集的准确率和F1,数据来自`get_test_data()`<br><br>
predict(time_step=args.time_step)<br>
用于预测第二天的收盘价,数据来自`get_predict_data(args.predict_dir)`<br><br>

    5、stock_main.py 主函数
可以调用上述所有函数接口，实现相关功能。<br><br>
## 相关论文
《基于LSTM的股票价格的多分类预测》<br><br> 
论文地址：https://www.hanspub.org/journal/PaperInformation.aspx?paperID=32542<br><br>
