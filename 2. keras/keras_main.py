import os, sys
import argparse
import time
import random
import cv2
import numpy as np

from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import efficientnet.keras as efn

import nsml
# from nsml.constants import DATASET_PATH, GPU_NUM

DATASET_PATH = '../sample_image'
IMSIZE = 280, 280
VAL_RATIO = 0.1
RANDOM_SEED = 1234

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded!')

    def infer(data):            # test mode
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        X = ImagePreprocessing(data)
        X = np.array(X)
        X = np.expand_dims(X, axis=-1)
        pred = model.predict_classes(X)  # 모델 예측 결과: 0-3
        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)


def Class2Label(cls):
    lb = [0] * 4
    lb[int(cls)] = 1
    return lb

def DataLoad(imdir):
    impath = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(imdir) for f in files if all(s in f for s in ['.jpg'])]
    img = []
    lb = []
    print('Loading', len(impath), 'images ...')
    for i, p in enumerate(impath):
        img_whole = cv2.imread(p, 0)
        h, w = img_whole.shape
        h_, w_ = h, w//2
        l_img = img_whole[:, w_:2*w_]
        r_img = img_whole[:, :w_]
        _, l_cls, r_cls = os.path.basename(p).split('.')[0].split('_')
        if l_cls=='0' or l_cls=='1' or l_cls=='2' or l_cls=='3':
            img.append(l_img);      lb.append(Class2Label(l_cls))
        if r_cls=='0' or r_cls=='1' or r_cls=='2' or r_cls=='3':
            img.append(r_img);      lb.append(Class2Label(r_cls))
    print(len(img), 'data with label 0-3 loaded!')
    return img, lb

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

    return wb_cl

def ImagePreprocessing(img):
    print('Preprocessing ...')
    final = img.copy()
    for i, im, in enumerate(img):
        tmp = im.copy()

        # Cropping
        h, w = tmp.shape[:2]
        h_ = h // 3
        tmp = tmp[h_:h_ * 2, :]

        # Resizing
        tmp = cv2.resize(tmp, dsize=(IMSIZE[1], IMSIZE[0]), interpolation=cv2.INTER_AREA)
        tmp = tmp / 255.
        resize = tmp

        # clahe + WB
        claheWB = WB_clahe(resize)

        # contour
        _, threshold = cv2.threshold(claheWB, 127, 255, 0)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours(claheWB)

        concat = np.concatenate([resize, claheWB, contour], axis=-1)
        final[i] = concat

    print(len(final), 'images processed!')
    return final

def shuffle_split_data(X, y, train_percent):
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

    return X_train, Y_train, X_val, Y_val

def transform_image(img, ang_range, shear_range, trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.
    A Random uniform distribution is used to generate different parameters for transformation
    '''
    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))
    return img

def f1score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def GetF1score(y, y_pred, target):
    tp = 0; fp = 0; fn = 0
    for i, y_hat in enumerate(y_pred):
        if (y[i] == target) and (y_hat == target):
            tp += 1
        if (y[i] == target) and (y_hat != target):
            fn += 1
        if (y[i] != target) and (y_hat == target):
            fp += 1
    try:
        f1s = tp / (tp + (fp + fn) / 2)
    except ZeroDivisionError:
        f1s = 0
    return f1s

def weightedf1score(y, y_pred, num_classes):
    F1scores = []
    for t in range(num_classes):
        F1scores.append(GetF1score(y, y_pred, str(t)))
    WeightedF1score = sum([(i+1)*f1 for i, f1 in enumerate(F1scores)])/10
    print('Weighted F1Score : {}'.format(WeightedF1score))

def EfficientNet_(input_shapes, num_classes):
    efficient_net = efn.EfficientNetB4(include_top=False, weights="imagenet")
    # x = efficient_net.output

    image = Input((IMSIZE[0], IMSIZE[1], 3), name='RGB')
    x = efficient_net(image).output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_initializer=TruncatedNormal(stddev=0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_initializer=TruncatedNormal(stddev=0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_initializer=TruncatedNormal(stddev=0.01))(x)
    pred = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=efficient_net.input, outputs=pred)
    return model


def ParserArguments(args):
    # Setting Hyperparameters
    args.add_argument('--epoch', type=int, default=100)          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=4)      # batch size 설정
    args.add_argument('--learning_rate', type=float, default=1e-4)
    args.add_argument('--learning_rate_decay', type=float, default=1e-5)# learning rate 설정
    args.add_argument('--num_classes', type=int, default=4)     # 분류될 클래스 수는 4개

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()
    return config.epoch, config.batch_size, config.num_classes, config.learning_rate, config.learning_rate_decay, config.pause, config.mode


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    nb_epoch, batch_size, num_classes, learning_rate, learning_rate_decay, ifpause, ifmode = ParserArguments(args)

    seed = 1234
    np.random.seed(seed)

    """ Model """
    h, w = IMSIZE
    model = EfficientNet_((h, w, 3), 4)
    adam = optimizers.Adam(lr=learning_rate, decay=learning_rate_decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1score])
    bind_model(model)

    if ifpause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if ifmode == 'train':  ### training mode일 때
        print('Training Start...')
        images, labels = DataLoad(os.path.join(DATASET_PATH, 'train'))
        images = ImagePreprocessing(images)
        ## data 섞기
        images = np.array(images)
        labels = np.array(labels)
        dataset = [[X, Y] for X, Y in zip(images, labels)]
        random.shuffle(dataset)

        X = np.array([n[0] for n in dataset])
        Y = np.array([n[1] for n in dataset])


        ## Augmentation 예시
        kwargs = dict(
            rotation_range=10,
            zoom_range=5,
            width_shift_range=0.0,
            height_shift_range=10.0,
            horizontal_flip=True,
            vertical_flip=False
        )

        # then flow and fit_generator....

        """ Callback """
        monitor = 'f1score'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        STEP_SIZE_TRAIN = len(X) // batch_size
        print('\n\nSTEP_SIZE_TRAIN = {}\n\n'.format(STEP_SIZE_TRAIN))

        ## data를 trainin과 validation dataset으로 나누기
        X_train, Y_train, X_val, Y_val = shuffle_split_data(X, Y, 100*(1-VAL_RATIO))

        train_datagen = ImageDataGenerator(**kwargs)
        train_generator = train_datagen.flow(x=X_train, y=Y_train, shuffle=False, batch_size=batch_size, seed=seed)

        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            print("### Model Fitting.. ###")
            print('epoch = {} / {}'.format(epoch + 1, nb_epoch))
            print('check point = {}'.format(epoch))

            # for no augmentation case
            hist = model.fit_generator(generator=train_generator,
                                       validation_data=(X_val, Y_val),
                                       batch_size=batch_size,
                                       callbacks=[reduce_lr],
                                       shuffle=True,
                                       class_weight={0:0.32, 1:1.92, 2:4.17, 3:8.33})
            pred = model.predict(X_val)
            score = weightedf1score(pred, Y_val, num_classes)

            print(hist.history)
            train_acc = hist.history['f1score'][0]
            train_loss = hist.history['loss'][0]
            val_acc = hist.history['val_f1score'][0]
            val_loss = hist.history['val_loss'][0]
            nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc, val_loss=val_loss, val_acc=val_acc)
            nsml.save(epoch)
            print('Training time for one epoch : %.1f' % (time.time() - t1))
        print('Total training time : %.1f' % (time.time() - t0))
        print('Total training time : %.1f' % (time.time() - t0))