import pandas as pd


def get_data_len(file_path):
    if file_path:
        with open(file_path) as f:
            df = pd.read_csv(f)
            return len(df)


# -------------------参数配置----------------- #
class Arg:
    def __init__(self):
        # 训练集数据存放路径
        self.train_dir = './data/train_mix-17-18.csv'
        # 测试集数据存放路径
        self.test_dir = './data/test_mix.csv'
        # 更新数据存放路径
        self.new_dir = './data/train_mix-19.csv'
        # 要预测的数据存放路径
        self.predict_dir = './data/000001.csv'
        # 模型存放路径
        self.train_model_dir = './model/'
        # fining-turn模型存放路径
        self.fining_turn_model_dir = './data/finet/'
        # 训练图存放路径
        self.train_graph_dir = './data/graph/train_270/'
        # 验证loss存放路径
        self.val_graph_dir = './data/graph/val_270/'
        # 模型名称
        self.model_name = 'model-270-17-19'
        self.model_name_ft = 'model-ft-01-03'
        self.rnn_unit = 128  # 隐层节点数
        self.input_size = 6  # 输入维度（既用几个特征）
        self.output_size = 6  # 输出维度（既使用分类类数预测）
        self.layer_num = 3  # 隐藏层层数
        self.lr = 0.0006  # 学习率
        self.time_step = 20  # 时间步长
        self.epoch = 50  # 训练次数
        self.epoch_fining = 30  # 微调的迭代次数
        # 单只股票的长度（同一数据集股票长度应处理等长）
        self.stock_len = get_data_len('./data/399300.csv')
        # 更新后单只股票的长度（同一数据集股票长度应处理等长）
        self.stock_len_new = get_data_len('./data/399300.csv')
        self.batch_size = 1024  # batch_size
        self.ratio = 0.8  # 训练集验证集比例
