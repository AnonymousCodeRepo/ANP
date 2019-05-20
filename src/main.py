from __future__ import print_function
import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR
import os
from utils import *
from LeNet import *
from ResNet_16 import ResNet50
from resnet18 import ResNet18
from VGG import VGG16
from VGG13 import VGG13
from VGG11 import VGG11
from VGG19 import VGG19
from denseNet import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

from torch.autograd import Variable

def get_bool(string):
    if(string == 'False'):
        return False
    else:
        return True

# Training settings
parser = argparse.ArgumentParser(description='lat implementation')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--epoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
parser.add_argument('--alpha', type=float, default=0.6, help='alpha')
parser.add_argument('--epsilon', type=float, default=0.6, help='epsilon')
parser.add_argument('--enable_lat', type=get_bool, default=True, help='enable anp training')
parser.add_argument('--layerlist', default="0,1", help='layerlist of lat')
parser.add_argument('--test_flag', type=get_bool, default=True, help='test or train')
parser.add_argument('--adv_flag', type=get_bool, default=False, help='adv or clean')
parser.add_argument('--test_data_path', default="", help='test data path')
parser.add_argument('--test_label_path', default="", help='test label path')
parser.add_argument('--train_data_path', default="", help='training dataset path')
parser.add_argument('--model_path', default="", help='number of classes')
parser.add_argument('--pro_num', type=int, default=8, help='progressive number')
parser.add_argument('--batchnorm', type=get_bool, default=True, help='batch normalization')
parser.add_argument('--dropout', type=get_bool, default=True, help='dropout')
parser.add_argument('--dataset', default='mnist', help='data set')
parser.add_argument('--model', default='lenet', help='target model, [lenet, resnet, vgg, ...]')
parser.add_argument('--logfile',default='log.txt',help='log file to accord validation process')
parser.add_argument('--enable_noise', type=get_bool, default=False, help='enable gaussian noise on gradient')
parser.add_argument('--loss_acc_path',default='./loss_acc/train_loss/',help='save train loss as .p to draw pic')
#parser.add_argument('--test_loss_acc_path',default='./loss_acc/train_acc/',help='save train acc as .p to draw pic')


args = parser.parse_args()
#print(args)


def train_op(model):
    f=open(args.logfile,'w')
    # load training data and test set
    if args.dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(
            root=args.train_data_path,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=False
        )
        test_data = torchvision.datasets.MNIST(
            root=args.train_data_path,
            train=False)

    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()])
        train_data = torchvision.datasets.CIFAR10(
            root=args.train_data_path,
            train=True,
            transform=transform,
            download=False
        )
        test_data = torchvision.datasets.CIFAR10(
            root=args.train_data_path,
            train=False)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)

    if args.dataset == 'mnist':
        test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:args.batchsize].cuda() / 255.
        test_y = test_data.test_labels[:args.batchsize].cuda()
    if args.dataset == 'cifar10':
        test_x = torch.Tensor(test_data.test_data).view(-1,3,32,32)[:args.batchsize].cuda() / 255.
        test_y = torch.Tensor(test_data.test_labels)[:args.batchsize].cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay = 5e-4)
    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 60
    # lr = 0.0005   if 60 <= epoch < 90
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()
    curr_lr = args.lr
    
    train_losses = []
    train_acc = []
    test_acc = []
    train_step = []
    test_step = []
    for epoch in range(args.epoch):
        #adjust_learning_rate(args.lr, optimizer, epoch)
        #---------------------------------------------------------
        if args.enable_lat:
            args.epsilon, args.alpha, args.pro_num = set_anp(epoch)
            model.update_anp(args.epsilon, args.alpha, args.pro_num)
        #----------------------------------------------------------
        
        scheduler.step()
        for step, (x, y) in enumerate(train_loader):
			# enable anp training
            if not args.enable_lat:
                args.pro_num = 1
            if not len(y) == args.batchsize:
                continue
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
			# enable anp training
            if args.enable_lat:
                model.zero_reg()
            # progressive process
            for iter in range(args.pro_num):
                iter_input_x = b_x
                iter_input_x.requires_grad = True
                iter_input_x.retain_grad()

                logits = model(iter_input_x)
                loss = loss_func(logits, b_y)
                optimizer.zero_grad()
                loss.backward()
                #print(model.z0.grad.size())
                #nn.utils.clip_grad_norm_(model.parameters(),args.batchsize)
                optimizer.step()
                model.save_grad()               

            # test acc for validation set
            if step % 100 == 0:
                if args.enable_lat:
                    model.zero_reg()
                f.write('[Epoch={}/{}]: step={}/{},'.format(epoch,args.epoch,step,len(train_loader)))
                print('epoch={}/{}, step={}/{}'.format(epoch,args.epoch,step,len(train_loader)))
                acc = test_op(model,f)
                test_acc.append(acc)
                test_step.append(epoch*len(train_loader) + step)
                

            # save model
            if step % 200 == 0 and step != 0:
                if args.enable_lat:
                    test_adv(model,0.031,f)
                    test_adv(model,0.063,f)
                print('saving model...')
                print('lat={}, pro/eps/a={}/{}/{}'.format(args.enable_lat, args.pro_num, args.epsilon, args.alpha))
				# enable anp training
                if args.enable_lat:
                    torch.save(model.state_dict(), args.model_path + 'lat_param.pkl')
                else:
                    torch.save(model.state_dict(), args.model_path + 'naive_param.pkl')


            
            # print batch-size predictions from training data
            if step % 20 == 0:
                if args.enable_lat:
                    model.zero_reg()
                
                with torch.no_grad():
                    test_output = model(b_x)
                train_loss = loss_func(test_output, b_y)
                pred_y = torch.max(test_output, 1)[1].cuda().data.cpu().squeeze().numpy()
                Accuracy = float((pred_y == b_y.data.cpu().numpy()).astype(int).sum()) / float(b_y.size(0))
                print('train loss: %.4f' % train_loss.data.cpu().numpy(), '| train accuracy: %.2f' % Accuracy)
                train_losses.append(train_loss.item())
                train_acc.append(Accuracy)
                test_step.append(epoch*len(train_loader) + step)

        save_loss_acc(args.loss_acc_path,train_losses,train_acc,test_acc,train_step,test_step)        
        #end for batch
        
    # end for epoch
    
    f.close()


def test_op(model,f=None):
    # get test_data , test_label from .p file
    test_data, test_label, size = read_data_label(args.test_data_path,args.test_label_path)

    if size == 0:
        print("reading data failed.")
        return
    
    #data = torch.from_numpy(data).cuda()
    #label = torch.from_numpy(label).cuda()

    test_data = test_data.cuda()
    test_label = test_label.cuda()
    
    # create dataset
    testing_set = Data.TensorDataset(test_data, test_label)

    testing_loader = Data.DataLoader(
        dataset=testing_set,
        batch_size=args.batchsize, # without minibatch cuda will out of memory
        shuffle=False,
        #num_workers=2
        drop_last=True,
    )
    
    # Test the model
    model.eval()
    correct = 0
    total = 0
    for x, y in testing_loader:
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            h = model(x)
        _, predicted = torch.max(h.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    acc = 100 * correct / total
    print('Accuracy of the model on the test images: {:.2f} %'.format(acc))        
    if f != None:
        f.write('Accuracy on the test images: {:.2f} %'.format(acc))
        f.write('\n')
    #print('now is {}'.format(type(model)))
    model.train(True)
    return acc

def test_adv(model,eps,f=None):
    # get test_data , test_label from .p file
    if eps == 0.031:
        data_path = "/test_adv(eps_0.031).p"
    elif eps == 0.063:
        data_path = "/test_adv(eps_0.063).p"
    label_path = "/test_label.p"
    
    test_data, test_label, size = read_data_label(data_path,label_path)

    if size == 0:
        print("reading data failed.")
        return
    
    #data = torch.from_numpy(data).cuda()
    #label = torch.from_numpy(label).cuda()

    test_data = test_data.cuda()
    test_label = test_label.cuda()
    
    # create dataset
    testing_set = Data.TensorDataset(test_data, test_label)

    testing_loader = Data.DataLoader(
        dataset=testing_set,
        batch_size=args.batchsize, # without minibatch cuda will out of memory
        shuffle=False,
        #num_workers=2
        drop_last=True,
    )
    
    # Test the model
    model.eval()
    correct = 0
    total = 0
    for x, y in testing_loader:
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            h = model(x)
        _, predicted = torch.max(h.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    print('Accuracy of the model on the eps_{} images: {:.2f} %'.format(eps ,(100 * correct / total)) )        
    if f != None:
        f.write('Accuracy of the model on the eps_{} images: {:.2f} %'.format(eps ,(100 * correct / total)) )
        f.write('\n')
    #print('now is {}'.format(type(model)))
    model.train(True)

def test_one(model,data_cat,data_path):
    
    label_path = "/test_label.p"
    test_data, test_label, size = read_data_label(data_path,label_path)

    if size == 0:
        print("reading data failed.")
        return
    
    test_data = test_data.cuda()
    test_label = test_label.cuda()
    
    # create dataset
    testing_set = Data.TensorDataset(test_data, test_label)

    testing_loader = Data.DataLoader(
        dataset=testing_set,
        batch_size=args.batchsize, # without minibatch cuda will out of memory
        shuffle=False,
        #num_workers=2
        drop_last=True,
    )
    
    # Test the model
    model.eval()
    correct = 0
    total = 0
    for x, y in testing_loader:
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            h = model(x)
        _, predicted = torch.max(h.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    print('Model Acc on {} : {:.2f} %'.format( data_cat,(100 * correct / total)) )        
    model.train(True)

def test_all(model):

    model_list = ['vgg','resnet','densenet','inception']
    adv_data_vgg = {
    'fgsm-e8-vgg':"/test_adv(eps_0.031).p",
    'fgsm-e16-vgg':"/test_adv(eps_0.063).p",
    'stepll-e8-vgg':"/test_stepll/test_adv(eps_0.031).p",
    'stepll-e16-vgg':"/test_stepll/test_adv(eps_0.063).p",
    'pgd-a16-vgg':"/test_pgd/test_adv(eps_0.031).p",
    'pgd-a2-vgg':"/test_pgd/test_adv(eps_0.03).p",
    'momentum-e8-vgg':"/test_momentum/vgg/test_adv(eps_0.031).p",
    }
    adv_data_resnet = {
    'fgsm-e8-resnet':"/test/resnet/test_adv(eps_0.031).p",
    'fgsm-e16-resnet':"/test/resnet/test_adv(eps_0.063).p",
    'stepll-e8-resnet':"/test_stepll/resnet/test_adv(eps_0.031).p",
    'stepll-e16-resnet':"/test_stepll/resnet/test_adv(eps_0.063).p",
    'pgd-a16-resnet':"/test_pgd/resnet/test_adv(eps_0.031_a_0.063).p",
    'pgd-a2-resnet':"/test_pgd/resnet/test_adv(eps_0.031_a_0.008).p",
    'momentum-e8-resnet':"/test_momentum/resnet/test_adv(eps_0.031).p",
    }
    adv_data_densenet = {
    'fgsm-e8-densenet':"/test/densenet/test_adv(eps_0.031).p",
    'fgsm-e16-densenet':"/test/densenet/test_adv(eps_0.063).p",
    'stepll-e8-densenet':"/test_stepll/densenet/test_adv(eps_0.031).p",
    'stepll-e16-densenet':"/test_stepll/densenet/test_adv(eps_0.063).p",
    'pgd-a16-densenet':"/test_pgd/densenet/test_adv(eps_0.031_a_0.063).p",
    'pgd-a2-densenet':"/test_pgd/densenet/test_adv(eps_0.031_a_0.008).p",
    'momentum-e8-densenet':"/test_momentum/densenet/test_adv(eps_0.031).p",
    }
    adv_data_inception = {
    'fgsm-e8-inception':"/test/inception/test_adv(eps_0.031).p",
    'fgsm-e16-inception':"/test/inception/test_adv(eps_0.063).p",
    'stepll-e8-inception':"/test_stepll/inception/test_adv(eps_0.031).p",
    'stepll-e16-inception':"/test_stepll/inception/test_adv(eps_0.063).p",
    'pgd-a16-inception':"/test_pgd/inception/test_adv(eps_0.031_a_0.063).p",
    'pgd-a2-inception':"/test_pgd/inception/test_adv(eps_0.031_a_0.008).p",
    'momentum-e8-inception':"/test_momentum/inception/test_adv(eps_0.031).p",   
    }
    
    print('Now is clean data')
    test_op(model)
    
    for target in model_list:
        print('Now adv data come from {}'.format(target))
  
        if target == 'vgg':
            data_list = adv_data_vgg
        elif target == 'resnet':
            data_list = adv_data_resnet
        elif target == 'densenet':
            data_list = adv_data_densenet
        elif target == 'inception':
            data_list = adv_data_inception
  
        for data_cat in data_list:
            data_path = data_list[data_cat]
            test_one(model,data_cat,data_path)
                    
def choose_model():
    # switch models
    print(args.model)
    print(args.layerlist)
    if args.model == 'lenet':
        cnn = LeNet(enable_lat=args.enable_lat,
                    layerlist=args.layerlist,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    batch_norm=args.batchnorm,
                    if_dropout=args.dropout
                    )
    elif args.model == 'resnet':
        cnn = ResNet50(enable_lat=args.enable_lat,
                    layerlist=args.layerlist,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    if_dropout=args.dropout
                    )
    elif args.model == 'resnet18':
        cnn = ResNet18(enable_lat=args.enable_lat,
                    layerlist=args.layerlist,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    if_dropout=args.dropout
                    )
    elif args.model == 'vgg16':
        cnn = VGG16(enable_lat=args.enable_lat,
                    layerlist=args.layerlist,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    if_dropout=args.dropout
                    )
    elif args.model == 'vgg11':
        cnn = VGG11(enable_lat=args.enable_lat,
                    layerlist=args.layerlist,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    if_dropout=args.dropout
                    )
    elif args.model == 'vgg13':
        cnn = VGG13(enable_lat=args.enable_lat,
                    layerlist=args.layerlist,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    if_dropout=args.dropout
                    )
    elif args.model == 'vgg19':
        cnn = VGG19(enable_lat=args.enable_lat,
                    layerlist=args.layerlist,
                    epsilon=args.epsilon,
                    alpha=args.alpha,
                    pro_num=args.pro_num,
                    batch_size=args.batchsize,
                    if_dropout=args.dropout
                    )
    elif args.model == 'densenet':
        cnn = DenseNet()
        
    cnn.cuda()
    if args.enable_lat:
        cnn.choose_layer()

    return cnn

if __name__ == "__main__":

    if os.path.exists(args.model_path) == False:
        os.makedirs(args.model_path)
		
	# enable anp training
    if args.enable_lat:
        real_model_path = args.model_path + "lat_param.pkl"
        print('loading the LAT model')
    else:
        real_model_path = args.model_path + "naive_param.pkl"
        print('loading the naive model')
    
    
    if os.path.exists(args.loss_acc_path) == False:
        os.makedirs(args.loss_acc_path)

    args.test_flag = False
    if args.test_flag:
        args.enable_lat = False
    
    cnn = choose_model()

    if os.path.exists(real_model_path):
        cnn.load_state_dict(torch.load(real_model_path))
        print('load model.')
    else:
        print("load failed.")

    if args.test_flag:
        if args.adv_flag:
            test_all(cnn)
        else:
            test_op(cnn)
    else:
        train_op(cnn)
        
