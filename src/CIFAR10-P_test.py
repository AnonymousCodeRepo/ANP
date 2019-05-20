#--------------------------------------------------------
# perturbation evaluation on CIFAR10-P
#--------------------------------------------------------
import torch
import numpy as np
import torchvision
from VGG import *
import os
import argparse



#calculate all kinds of perturbation's FP of one model 
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='CIFAR10-C test')
    parser.add_argument('--batchsize', type=int, default=64, help='model batch size')
    parser.add_argument('--model_name', default="ANP", help='model\'s name')
    parser.add_argument('--model_path', default="/model_path/VGG.pkl", help='model path')
    parser.add_argument('--distotion_root', default="/distotion_root/", help='the path of the folder which contains all kinds of distotions.npy')
    args = parser.parse_args()

    Noise_distortion = ['gaussian_noise','shot_noise','speckle_noise']
    Other_distortion = ['brightness','gaussian_blur','motion_blur','rotate','scale','shear','snow','spatter','tilt','translate','zoom_blur']

    #load model
    net = VGG16(enable_lat = False,
                epsilon = 0,
                pro_num = 1,
                batch_size = args.batchsize,
                if_dropout = True)
    net.cuda()
    net.load_state_dict(torch.load(args.model_path))
    
    for i in range(len(Noise_distortion) + len(Other_distortion)):                
        if(i >= len(Noise_distortion)):
            distortion_name = Other_distortion[i - len(Noise_distortion)]
        else:
            distortion_name = Noise_distortion[i]
        data_root = args.distotion_root + distortion_name + '.npy'
        if('noise' in distortion_name):
            noise_flag = True
        else:
            noise_flag = False
    
        data = np.load(data_root)
        data_num = data.shape[0]
        one_batch = data.shape[1] - 1

        diff_pred = 0
        for item in range(data_num):
            x = data[item]
            x = x.transpose((0,3,1,2))
            x = x/255.0
            test_x = torch.from_numpy(x).float().cuda()
            with torch.no_grad():
                h = net(test_x)
            predicted = torch.max(h.data, 1)[1].to('cpu').numpy()
            prev_pred = predicted[0]
            for pred in predicted:
                diff_pred += int(pred != prev_pred)
                if not noise_flag:
                    prev_pred = pred
        FR = 100.0 * float(diff_pred) / (data_num * one_batch)
        print(args.model_name + ' ' + distortion_name + '  ' + str(noise_flag) + '  ' + '  FP(%) : {:.2f}'.format(FR))
            



