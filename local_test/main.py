import os, sys
import argparse
import time
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from f1score import *
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import *
from PIL import Image

import nsml
# from nsml.constants import DATASET_PATH, GPU_NUM

DATASET_PATH = './sample_image/'
IMSIZE = 224, 224
VAL_RATIO = 0.2
RANDOM_SEED = 1234


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')

    def infer(data):  ## test mode
        X = ImagePreprocessing(data)
        X = np.array(X)
        X = np.expand_dims(X, axis=1)
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(device)
            pred = model.forward(X)
            prob, pred_cls = torch.max(pred, 1)
            pred_cls = pred_cls.tolist()
        print('Prediction done!\n Saving the result...')
        return pred_cls

    nsml.bind(save=save, load=load, infer=infer)


def DataLoad(imdir):
    impath = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(imdir) \
              for f in files if all(s in f for s in ['.jpg'])]
    img = []
    lb = []
    print('Loading', len(impath), 'images ...')
    for i, p in enumerate(impath):
        img_whole = cv2.imread(p, 0)
        h, w = img_whole.shape
        h_, w_ = h, w//2
        l_img = img_whole[:, w_:2 * w_]
        r_img = img_whole[:, :w_]
        _, l_cls, r_cls = os.path.basename(p).split('.')[0].split('_')
        if l_cls=='0' or l_cls=='1' or l_cls=='2' or l_cls=='3':
            img.append(l_img); lb.append(int(l_cls))
        if r_cls=='0' or r_cls=='1' or r_cls=='2' or r_cls=='3':
            img.append(r_img); lb.append(int(r_cls))
    print(len(img), 'data with label 0-3 loaded!')
    return img, lb


def ImagePreprocessing(imgs):
    print('Preprocessing ...')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    for i, im, in enumerate(imgs):
        # Applying clahe
        tmp = clahe.apply(im)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

        # Cropping
        h, w = tmp.shape[:2]
        h_ = h // 3
        tmp = tmp[h_:h_ * 2, :]

        # Resizing
        tmp = cv2.resize(tmp, dsize=(IMSIZE[1], IMSIZE[0]), interpolation=cv2.INTER_AREA)
        tmp = tmp / 255.
        imgs[i] = tmp

    print(len(imgs), 'images processed!')
    return imgs

def ParserArguments(args):
    # Setting Hyper parameters
    args.add_argument('--epoch', type=int, default=100)          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=8)      # batch size 설정
    args.add_argument('--learning_rate', type=float, default=0.001)  # learning rate 설정
    args.add_argument('--learning-rate-decay', type=float, default=0.9) # learning rate decay 설
    args.add_argument('--num_classes', type=int, default=4)     # 분류될 클래스 수는 4개

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()
    return config.epoch, config.batch_size, config.num_classes, config.learning_rate, config.learning_rate_decay, \
           config.pause, config.mode




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    nb_epoch, batch_size, num_classes, learning_rate, learning_rate_decay, ifpause, ifmode = ParserArguments(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #####   Model   #####
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=4)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate_decay)
    bind_model(model)

    if ifpause:  ## for test mode
        print('Inferring Start ...')
        nsml.paused(scope=locals())

    if ifmode == 'train':  ## for train mode
        print('Training start ...')
        images, labels = DataLoad(imdir=os.path.join(DATASET_PATH, 'train'))
        images, labels = ImagePreprocessing(images, labels)
        images = np.array(images)
        images = np.transpose(images, (0,3,1,2))
        labels = np.array(labels)


        dataset = TensorDataset(torch.from_numpy(images).float(), torch.from_numpy(labels).long())
        subset_size = [len(images) - int(len(images) * VAL_RATIO),int(len(images) * VAL_RATIO)]
        tr_set, val_set = random_split(dataset, subset_size)
        transformed_tr_set = Dataset()
        batch_train = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
        batch_val = DataLoader(val_set, batch_size=1, shuffle=False)

        _, ratio = np.unique(labels, return_counts=True)
        normed_weights = [1 - (x / sum(ratio)) for x in ratio]
        normed_weights = torch.FloatTensor(normed_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=normed_weights)

        #####   Training loop   #####
        STEP_SIZE_TRAIN = len(images) // batch_size
        print('\n\nSTEP_SIZE_TRAIN= {}\n\n'.format(STEP_SIZE_TRAIN))
        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            print('Model fitting ...')
            print('epoch = {} / {}'.format(epoch + 1, nb_epoch))
            print('check point = {}'.format(epoch))
            a, a_val, tp, tp_val = 0, 0, 0, 0
            for i, (x_tr, y_tr) in enumerate(batch_train):
                x_tr, y_tr = x_tr.to(device), y_tr.to(device)
                optimizer.zero_grad()
                pred = model(x_tr)
                loss = criterion(pred, y_tr)
                loss.backward()
                optimizer.step()
                prob, pred_cls = torch.max(pred, 1)
                a += y_tr.size(0)
                tp += (pred_cls == y_tr).sum().item()

            with torch.no_grad():
                for j, (x_val, y_val) in enumerate(batch_val):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    pred_val = model(x_val)
                    loss_val = criterion(pred_val, y_val)
                    prob_val, pred_cls_val = torch.max(pred_val, 1)
                    a_val += y_val.size(0)
                    tp_val += (pred_cls_val == y_val).sum().item()

            acc = tp / a
            acc_val = tp_val / a_val
            print("  * loss = {}\n  * acc = {}\n  * loss_val = {}\n  * acc_val = {}".format(loss.item(), acc, loss_val.item(), acc_val))
            nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=loss.item(), acc=acc, val_loss=loss_val.item(), val_acc=acc_val)
            nsml.save(epoch)
            print('Training time for one epoch : %.1f\n' % (time.time() - t1))
        print('Total training time : %.1f' % (time.time() - t0))