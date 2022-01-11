log_path = './log/log.txt' #日志路径
root = '' #data根路径，使用相对路径所以这里设置为空
resume = 0 #是否测试

gpu = 2 #gpu序号
num_classes = 101 #种类个数
lr = 0.1 #学习率，SGD参数
batch_size = 128
weight_decay = 2e-4 #SGD参数
num_epochs = 75 #baseline 100 epoch: ~24%, 200 epoch: ~28%
momentum = 0.9 #SGD参数
cos = False
