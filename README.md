参数查找：运行param_search.py，得到该次combination中的最佳参数
训练模型：运行train.py，训练模型，保存模型参数和可视化图，包括训练集和验证集上的 loss 曲线和验证集上的 accuracy 曲线
设备为简陋的华为matebook14的老旧CPU，参数查找范围有限

第一次参数查找：
param_grid = {
    'lr': [0.03, 0.01],        # 学习率
    'hidden_size': [512, 1024],  # 隐藏层规模
    'reg': [0.01, 0.005],    # 正则化强度
    'batch_size': [128, 256]  # 批量大小
}
SEARCH_EPOCHS = 3
学习率固定衰减，
得到最佳参数：{'lr': 0.03, 'hidden_size': 1024, 'reg': 0.01, 'batch_size': 128}，准确率：0.4882
在此参数基础上，学习率乘1.2，批量乘2，以适配余弦退火算法。
用新参数训练模型，epochs=50，学习率余弦退火，验证集准确率为0.5660，测试集准确率为0.5586

第二次参数查找：
param_grid = {
    'lr': [0.1, 0.07, 0.05, 0.03],  # 增加0.07作为中间值
    'hidden_size': [896, 1024, 1152, 1280],  # 缩小范围，步长~128
    'reg': [0.02, 0.015, 0.012, 0.01],  # 细化正则化步长
    'batch_size': [128, 192, 256]  # 增加中间值192
}
SEARCH_EPOCHS = 5
由于睡觉时程序没能运行，只跑了前24个combination，
得到{'lr': 0.1, 'hidden_size': 1024, 'reg': 0.015, 'batch_size': 256}，准确率：0.5054
在此参数基础上，学习率乘1.2，批量乘2，以适配余弦退火算法。
用新参数训练模型，epochs=50，学习率余弦退火，验证集准确率为0.5608，测试集准确率为0.5618

第三次训练：
{'lr': 0.07, 'hidden_size': 1024, 'reg': 0.012, 'batch_size': 256}
在此参数基础上，学习率乘1.2，批量乘1.5，以适配余弦退火算法。
用新参数训练模型，epochs=50，学习率余弦退火，验证集准确率为0.5678，测试集准确率为0.5683

最终决定保留第三次训练参数，并将其作为最终参数。