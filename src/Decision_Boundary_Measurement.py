#--------------------------------------------------------
# empirical worst case decision boundary distance
#--------------------------------------------------------
import numpy as np
import os
import torch
from VGG import *
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import pickle
import argparse

#randomly choose some orthogonal vectors(returned vector size : vector_num * ic * iw * ih).
def get_orthogonal_vectors(data_path,vector_size,vector_num):
    if(os.path.isfile(data_path)):
        vectors = np.load(data_path)
    else:
        randmat = np.random.normal(size = (vector_size,vector_num))
        q,r = np.linalg.qr(randmat)
        vectors = q.T * np.sqrt(float(vector_size))
        np.save(data_path,vectors)
    return vectors

#make measurement of one image's decision boundriy.
def measure_one_img(model,img,label,vectors,vector_num,mags):
    model.eval()
    margin_list = []
    X = img.cuda()
    Y = label.cuda()
    with torch.no_grad():
        logits = model(X)
    _,original_prediction = torch.max(logits.data,1)
    if((original_prediction == Y).sum().item() == 0):
        return None
    num_mag = len(mags)
    for dir_index in range(vector_num):
        none_flag = True
        direction = torch.from_numpy(vectors[dir_index]).float().view(-1,ic,ih,iw)
        direction = direction.cuda()
        for i in  range(num_mag):
            mag = mags[i]
            new_X = X + mag * direction
            new_X = torch.clamp(new_X,imin,imax)
            with torch.no_grad():
                logits = model(new_X)
            _,new_prediction = torch.max(logits.data,1)
            if((new_prediction == Y).sum().item() == 0):
                margin_list.append(int(init_mags[i]))
                none_flag = False
                break
        if((i+1) == num_mag and none_flag):
            margin_list.append(255)
    return margin_list

#Calculate the empirical worst case decision boundary distance of 1000 iamges in the dataset.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Empirical Worst Case Decision Boundary Distance')
    parser.add_argument('--batchsize', type=int, default=64, help='model batch size')
    parser.add_argument('--vectors_path', default="../OrthogonalVectors.npy", help='Orthogonal Vectors path')
    parser.add_argument('--model_path', default="../VGG.pkl", help='model path')
    parser.add_argument('--dataset_path', default="./", help='dataset path')
    parser.add_argument('--margin_list_path', default="../VGG_margin_list.pkl", help='path of the result of the decision boundary distance list')
    args = parser.parse_args()
    
    #initial settting
    total_margin_list = []
    imin = 0.0
    imax = 1.0
    ih = 32
    iw = 32
    ic = 3
    isize = ic*iw*ih
    num_classes = 10
    vector_num = 1000 

    mags = np.arange(0,127.5,2,dtype = np.float32) 
    init_mags = mags.copy()
    mags = mags / 255.0
    mags = mags.tolist()
    
    test_data = torchvision.datasets.CIFAR10(
                    root = args.dataset_path,
                    train = False,
                    transform = transforms.ToTensor(),
                    download = False)
    test_loader = Data.DataLoader(dataset = test_data, batch_size = 1, shuffle = False)
    print('finish loading dataset')
    
    Vec = get_orthogonal_vectors(args.vectors_path,isize,vector_num)

    net = VGG16(enable_lat = False, epsilon = 0, pro_num = 1, batch_size = args.batchsize)
    net.cuda()
    net.load_state_dict(torch.load(args.model_path))
    print('finish loading model')
    
    count_num = 0
    for step,(x,y) in enumerate(test_loader):
        one_margin_list = measure_one_img(net,x,y,Vec,vector_num,mags)
        if(one_margin_list == None):
            continue
        else:
            total_margin_list.append(one_margin_list)
            count_num += 1
            print('finish ' + str(count_num) + ' example')
        if(count_num >= 100):
            break
    
    '''
    Saving the total_margin_list,the size of total_margin_list is 100 * 1000, which contains 1000 directions' decision boundary distance of 100 images.
    '''
    output = open(args.margin_list_path,'wb')
    pickle.dump(total_margin_list,output,-1)
    output.close() 
    
