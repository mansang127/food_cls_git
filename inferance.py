import config as cfg
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

def test(model):
    #加载模型
    path = 'ckpt/model_best.pth.tar'
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        #cfg.start_epoch = checkpoint['epoch']
        #best_prec = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict_model'])
        model = model.cuda(cfg.gpu)
    else:
        exit()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #关于CenterCrop的合理性 https://stackoverflow.com/questions/63856270/
        #pytorch-should-centercrop-be-used-to-test-set-does-this-count-as-cheating
        transforms.ToTensor(),
        normalize
    ])
    test_batch = 100
    test_path = "./data/food/test"
    files = os.listdir(test_path)
    test_list = torch.zeros(test_batch,3,224,224)
    pred_class = np.array([],int)
    for i in range(len(files)):
        print(i)
        if (i+1) % test_batch == 0:
            #每加载test_batch张图，一起进行预测
            test_list = test_list.cuda(cfg.gpu, non_blocking=True)
            model.eval()
            with torch.no_grad():
                if cfg.gpu is not None:
                    output = model(test_list)
                    _, predicted = output.max(1) #max(1)返回行最大索引
                    pred_class = np.append(pred_class, predicted.cpu().numpy())
            test_list = torch.zeros(test_batch,3,224,224)
        #加载一张图片
        with open(test_path + os.sep + files[i],'rb') as f:
            img = Image.open(f).convert('RGB')
            img = transform_test(img)
            #print(img,type(img),img.shape)
            test_list[i % test_batch] = img

    #写入文件
    dataframe = pd.DataFrame({'Id':files,'Expected':pred_class})
    dataframe.to_csv("./data/food/submission.csv",index=False,sep=',')
    # fo = open("./data/food/submission.txt", 'w')
    # fo.write('Id'+', '+ 'Expected'+ '\n')
    # for i in range(len(files)):
    #     info = files[i] + ', ' + str(pred_class[i]) + '\n'
    #     fo.write(info)
