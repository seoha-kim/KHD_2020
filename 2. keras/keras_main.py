import os, sys
import argparse
import time
import random
import cv2
import numpy as np

from keras import applications
from keras.layers import Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from f1score import WeightedF1Score
import efficientnet.keras as efn

import nsml
# from nsml.constants import DATASET_PATH, GPU_NUM

DATASET_PATH = '../3. local_test/sample_image/'
IMSIZE = 300, 300
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


def ImagePreprocessing(img):
    print('Preprocessing ...')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    for i, im, in enumerate(img):
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
        img[i] = tmp

    print(len(img), 'images processed!')
    return img

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

def EFFI(input_shapes, num_classes):
    efficient_net = efn.EfficientNetB4(weights='noisy-student', include_top=False, input_shape=(input_shapes[0], input_shapes[1], 1))
    x = efficient_net.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='relu')
    pred = Dense(num_classes, activation="softmax")(x)
    return pred


def ParserArguments(args):
    # Setting Hyperparameters
    args.add_argument('--epoch', type=int, default=10)          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=8)      # batch size 설정
    args.add_argument('--learning_rate', type=float, default=1e-4)  # learning rate 설정
    args.add_argument('--num_classes', type=int, default=4)     # 분류될 클래스 수는 4개

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()
    return config.epoch, config.batch_size, config.num_classes, config.learning_rate, config.pause, config.mode


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    nb_epoch, batch_size, num_classes, learning_rate, ifpause, ifmode = ParserArguments(args)

    seed = 1234
    np.random.seed(seed)

    """ Model """
    h, w = IMSIZE
    model = EFFI((h, w), 4)
    adam = optimizers.Adam(lr=learning_rate, decay=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy',WeightedF1Score])
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
        images = np.expand_dims(images, axis=-1)
        labels = np.array(labels)
        dataset = [[X, Y] for X, Y in zip(images, labels)]
        random.shuffle(dataset)

        X = np.array([n[0] for n in dataset])
        Y = np.array([n[1] for n in dataset])

        '''
        ## Augmentation 예시
        kwargs = dict(
            rotation_range=180,
            zoom_range=0.0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            horizontal_flip=True,
            vertical_flip=True
        )
        train_datagen = ImageDataGenerator(**kwargs)
        train_generator = train_datagen.flow(x=X_train, y=Y_train, shuffle= False, batch_size=batch_size, seed=seed)
        # then flow and fit_generator....
        '''

        """ Callback """
        monitor = 'categorical_accuracy'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        STEP_SIZE_TRAIN = len(X) // batch_size
        print('\n\nSTEP_SIZE_TRAIN = {}\n\n'.format(STEP_SIZE_TRAIN))

        ## data를 trainin과 validation dataset으로 나누기
        X_train, Y_train, X_val, Y_val = shuffle_split_data(X, Y, 100*(1-VAL_RATIO))

        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            print("### Model Fitting.. ###")
            print('epoch = {} / {}'.format(epoch + 1, nb_epoch))
            print('check point = {}'.format(epoch))

            # for no augmentation case
            hist = model.fit(X_train, Y_train,
                             validation_data=(X_val, Y_val),
                             batch_size=batch_size,
                             callbacks=[reduce_lr],
                             shuffle=True
                             )
            print(hist.history)
            train_acc = hist.history['categorical_accuracy'][0]
            train_loss = hist.history['loss'][0]
            val_acc = hist.history['val_categorical_accuracy'][0]
            val_loss = hist.history['val_loss'][0]
            train_f1 = hist.history[WeightedF1Score][0]
            val_f1 = hist.history[WeightedF1Score][0]
            nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc, val_loss=val_loss, val_acc=val_acc)
            nsml.save(epoch)
            print('Training time for one epoch : %.1f' % (time.time() - t1))
        print('Total training time : %.1f' % (time.time() - t0))