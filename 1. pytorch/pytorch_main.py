import os
import argparse
import time
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms

import nsml
from nsml.constants import DATASET_PATH, GPU_NUM

# DATASET_PATH = '../sample_image/'
IMSIZE = 224, 224
VAL_RATIO = 0.15
RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)


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
        X = np.transpose(X, (0, 3, 1, 2))
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

def shuffle_split_data(X, y, train_percent=80):
    # arr_rand = np.random.rand(X.shape[0])
    # y label별로 random하게 80%는 train, 20%는 test로
    y_classes = np.unique(y)
    for y_class in y_classes:
        all_location = np.where(y == y_class)[0]
        train_location = np.random.choice(all_location, np.max((int(len(all_location) * 0.8), 1)), replace=False)
        val_location = np.setdiff1d(all_location, train_location)
        y_train = y[train_location]; x_train = X[train_location]
        y_val = y[val_location]; x_val = X[val_location]
        if y_class == np.min(y_classes):
            X_train = x_train; Y_train = y_train
            X_val = x_val; Y_val = y_val
        else:
            X_train = np.concatenate((X_train, x_train))
            Y_train = np.concatenate((Y_train, y_train))
            X_val = np.concatenate((X_val, x_val))
            Y_val = np.concatenate((Y_val, y_val))

    print(len(X_train), len(Y_train), len(X_val), len(Y_val))
    return X_train, Y_train, X_val, Y_val

class PNSDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(np.uint8(image))
            image = self.transform(image)

        return image, label

def WB_clahe(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    r, g, b = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]

    # Find the gain of each channel
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg

    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

    balance_img = cv2.merge([b, g, r])
    ba_gr = cv2.cvtColor(balance_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    wb_cl = clahe.apply(ba_gr)
    final = cv2.cvtColor(wb_cl, cv2.COLOR_GRAY2RGB)

    return final

def ImagePreprocessing(img):
    print('Preprocessing ...')
    for i, im, in enumerate(img):
        # Applying clahe
        tmp = WB_clahe(im)

        # Cropping
        h, w = tmp.shape[:2]
        h_ = h // 3
        tmp = tmp[h_:h_ * 2, :]

        # Resizing
        tmp = cv2.resize(tmp, dsize=(IMSIZE[1], IMSIZE[0]), interpolation=cv2.INTER_AREA)
        tmp = tmp / 255.
        img[i] = tmp

    print(len(img), 'images processed!')
    return img


def ParserArguments(args):
    # Setting Hyper parameters
    args.add_argument('--epoch', type=int, default=100)          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=8)      # batch size 설정
    args.add_argument('--learning_rate', type=float, default=1e-4)  # learning rate 설정
    args.add_argument('--learning-rate-decay', type=float, default=0.01) # learning rate decay 설정
    args.add_argument('--num_classes', type=int, default=4)     # 분류될 클래스 수는 4개

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()
    return config.epoch, config.batch_size, config.num_classes, config.learning_rate, config.learning_rate_decay, \
           config.pause, config.mode

class EfficientNoisy(nn.Module):
    def __init__(self):
        super(EfficientNoisy, self).__init__()
        self.pretrained_model = EfficientNet.from_pretrained('efficientnet-b4')

    def forward(self, x):
        x = self.pretrained_model(x)

        x = nn.Linear(1000, 512)(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)

        x = nn.Linear(512, 128)(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(0.5)(x)

        x = nn.Linear(128, num_classes)(x)
        return x

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    nb_epoch, batch_size, num_classes, learning_rate, learning_rate_decay, ifpause, ifmode = ParserArguments(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #####   Model   #####
    model = EfficientNoisy()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate_decay)
    bind_model(model)

    if ifpause:  ## for test mode
        print('Inferring Start ...')
        nsml.paused(scope=locals())

    if ifmode == 'train':  ## for train mode
        print('Training start ...')
        images, labels = DataLoad(imdir=os.path.join(DATASET_PATH, 'train'))
        images = ImagePreprocessing(images)
        images = np.array(images)
        images = np.transpose(images, (0,3,1,2))
        labels = np.array(labels)

        image_transforms = {
        'train': transforms.Compose([transforms.ToPILImage(),
        			transforms.RandomRotation(degrees=(-5,5)),
        			transforms.RandomHorizontalFlip(),
        			transforms.RandomResizedCrop(224),
        			transforms.ToTensor(),
        			]),
        'val':transforms.Compose([
                    transforms.ToPILImage(),
        			transforms.ToTensor(),
                    ])
        }

        X_train, y_train, X_test, y_test = shuffle_split_data(images, labels)
        print('X_train: {} / X_val: {} / y_train: {} / y_val: {}'.format(X_train.shape, X_test.shape, y_train.shape,
                                                                         y_test.shape))

        # 1 - 7배 / 2 - 17배 / 3 - 20배
        for (num, increase) in [(0, 1),(1, 7),(2, 17),(3, 28)]:
            X_num = X_train[np.where(y_train == num)]
            y_num = y_train[np.where(y_train == num)]
            X_nums = np.tile(X_num, (increase,1,1,1))
            y_nums = np.tile(y_num, increase)
            if num == 0:
                final_X = X_nums
                final_y = y_nums
            else:
                final_X = np.concatenate((final_X, X_nums))
                final_y = np.concatenate((final_y, y_nums))
        print('final_X: {} / final_y: {}'.format(final_X.shape, final_y.shape))
        print('X_test: {} / y_test: {}'.format(X_test.shape, y_test.shape))

        tr_set = PNSDataset(final_X, final_y, transform=image_transforms['train'])
        val_set = PNSDataset(X_test, y_test, transform=image_transforms['val'])
        batch_train = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
        batch_val = DataLoader(val_set, batch_size=1, shuffle=False)

        criterion = nn.CrossEntropyLoss()


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