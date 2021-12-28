import xml.dom.minidom
import numpy as np
import pandas as pd
import sys
import os
from scipy.stats.mstats import winsorize
os.environ['TF_KERAS']='1'
os.putenv('TF_KERAS', '1')
os.environ.setdefault('TF_KERAS', '1')

import tensorflow as tf
from tensorflow import keras
from keras_transformer import get_encoders
from keras_transformer import gelu
from tensorflow.keras import backend as K

def FCN_bone(input_layber, features, ksize):
    conv1 = keras.layers.Conv1D(filters=features, kernel_size=ksize, padding='same')(input_layber)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)
    conv1 = keras.layers.MaxPool1D()(conv1)

    return  conv1

def Dense(shape):
    inputlayer = keras.layers.Input(shape)
    dense_layer = keras.layers.Dense(units=512, activation='relu')(inputlayer)
    #dense_layer = keras.layers.Dropout(0.3)(dense_layer)
    #dense_layer = keras.layers.Dense(units=128, activation='relu')(dense_layer)
    #dense_layer = keras.layers.Dropout(0.5)(dense_layer)
    dense_layer = keras.layers.Dense(units=256, activation='relu')(dense_layer)
    dense_layer = keras.layers.Dropout(0.5)(dense_layer)
    dense_layer = keras.layers.Dense(units=32, activation='relu')(dense_layer)

    outputlayer = keras.layers.Dense(1, activation='sigmoid')(dense_layer)
    model = keras.Model(inputs=inputlayer, outputs=outputlayer)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.AUC(name='auc'),
                           keras.metrics.Recall(name='Recall')
                         ])
    model.summary()
    return model
def FCN(shape):
    inputlayer = keras.layers.Input(shape)
    conv1 = FCN_bone(inputlayer, 48, 8)
    conv1 = FCN_bone(conv1, 64, 5)
    conv1 = FCN_bone(conv1, 128, 5)
    conv1 = FCN_bone(conv1, 256, 3)
    gap_layer = keras.layers.GlobalAveragePooling1D()(conv1)
    dense_layer = keras.layers.Dense(units=32, activation='relu')(gap_layer)
    outputlayer = keras.layers.Dense(1, activation='sigmoid')(dense_layer)
    model = keras.Model(inputs=inputlayer, outputs=outputlayer)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.AUC(name='auc'),
                           keras.metrics.Recall(name='Recall')
                         ])
    model.summary()
    return model
def Inception_bone(input_layer, n_feature_maps):
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(input_layer)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(input_layer)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    conv_z = keras.layers.Activation('relu')(conv_z)

    cnn_concat = keras.layers.concatenate([conv_x,conv_y, conv_z])

    return  cnn_concat
def Inception_cnn_1D(shape,  feature_num):
    n_feature_maps = feature_num
    input_layer = keras.layers.Input(name='the_input', shape=shape, dtype='float32')  # (None, 128, 64, 1)
    conv1 = FCN_bone(input_layer, 48, 8)
    conv1 = FCN_bone(conv1, 64, 5)
    incep_bone = Inception_bone(conv1, n_feature_maps)
    cnn_concat = keras.layers.Conv1D(filters=n_feature_maps*3, kernel_size=(3), padding='same')(incep_bone)
    cnn_concat = keras.layers.BatchNormalization()(cnn_concat)
    cnn_concat = keras.layers.Activation('relu')(cnn_concat)
    cnn_concat = keras.layers.MaxPool1D()(cnn_concat)

    gap_layer = keras.layers.GlobalAveragePooling1D()(cnn_concat)
    gap_layer = keras.layers.Dense(units=32, activation='relu')(gap_layer)
    # gap_layer = keras.layers.Dropout(dropout)(gap_layer)
    output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.AUC(name='auc'),
                           keras.metrics.Recall(name='Recall')
                         ])
    model.summary()
    return model
def Attention_encoder_model(shape):
    input_layer = keras.layers.Input(shape)
    conv1 = FCN_bone(input_layer, 64, 8)
    conv1 = FCN_bone(conv1, 128, 5)
    # conv block -1
    conv3 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
    conv3 = keras.layers.MaxPooling1D()(conv3)
    # split for attention
    attention_data = keras.layers.Lambda(lambda x: x[:, :, :128])(conv3)
    attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 128:])(conv3)
    # attention mechanism
    attention_softmax = keras.layers.Softmax()(attention_softmax)
    multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
    # last layer
    gap_layer = keras.layers.GlobalAveragePooling1D()(multiply_layer)
    dense_layer = keras.layers.Dense(units=32, activation='relu')(gap_layer)
    # output layer
    output_layer = keras.layers.Dense(units=1, activation='sigmoid')(dense_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.AUC(name='auc'),
                           keras.metrics.Recall(name='Recall')
                         ])
    model.summary()
    return model
def LSTM_Model(shape, feature_num):
    # Make Networkw
    inputs = keras.layers.Input(shape)

    conv1 = FCN_bone(inputs, 64, 5)
    # CNN to RNN
    lstm_1 = keras.layers.LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(
        conv1)  # (None, 32, 512)
    lstm_2 = keras.layers.LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm_1)#lstm1_merged
    # transforms RNN output to character activations:
    conv1 = FCN_bone(lstm_2, 256, 3)
    gap_layer = keras.layers.GlobalAveragePooling1D()(conv1)
    gap_layer = keras.layers.Dense(32,activation= "relu")(gap_layer)
    output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)  # (None, 32, 63)

    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.AUC(name='auc'),
                           keras.metrics.Recall(name='Recall')
                         ])
    model.summary()
    return model
def Attention(input_layer):
    conv1 = FCN_bone(input_layer, 64, 8)
    conv1 = FCN_bone(conv1, 128, 5)
    # conv block -1
    conv3 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
    conv3 = keras.layers.MaxPooling1D()(conv3)
    # split for attention
    attention_data = keras.layers.Lambda(lambda x: x[:, :, :128])(conv3)
    attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 128:])(conv3)
    # attention mechanism
    attention_softmax = keras.layers.Softmax()(attention_softmax)
    multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
    # last layer
    gap_layer = keras.layers.GlobalAveragePooling1D()(multiply_layer)
    dense_layer = keras.layers.Dense(units=32, activation='relu')(gap_layer)

    return dense_layer
def LSTM_Attention_Model(shape):
    # Make Networkw
    inputs = keras.layers.Input(shape)
    input_atten = keras.layers.Input(shape)
    atten_dense = Attention(input_atten)
    conv1 = FCN_bone(inputs, 64, 5)
    # CNN to RNN
    lstm_1 = keras.layers.LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(
        conv1)  # (None, 32, 512)
    lstm_2 = keras.layers.LSTM(128, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm_1)#lstm1_merged
    # transforms RNN output to character activations:
    conv1 = FCN_bone(lstm_2, 256, 3)
    gap_layer = keras.layers.GlobalAveragePooling1D()(conv1)
    gap_layer = keras.layers.Dense(32,activation= "relu")(gap_layer)
    contac_layer = keras.layers.add([gap_layer,atten_dense])
    #contac_layer = keras.layers.concatenate([gap_layer,atten_dense], axis=-1)
    dense = keras.layers.Dense(8, activation= 'relu')(contac_layer)
    output_layer = keras.layers.Dense(1, activation='sigmoid')(dense)  # (None, 32, 63)

    model = tf.keras.models.Model(inputs=[inputs,input_atten], outputs=output_layer)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[#keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.AUC(name='auc'),
                           keras.metrics.Recall(name='Recall')
                         ])
    model.summary()
    return model

def residual_bone(input_layer,ksize, feature_num,is_pool):
    n_feature_maps = feature_num
    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, padding='same', kernel_size=ksize)(input_layer)#
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps, padding='same', kernel_size=ksize)(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps, padding='same', kernel_size=ksize)(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block = keras.layers.add([shortcut_y, conv_z])
    output_block = keras.layers.Activation('relu')(output_block)
    if is_pool:
        output_block = keras.layers.MaxPool1D()(output_block)
    # BLOCK 2

    return output_block

def Resnet_model(shape):
    input_layer = keras.layers.Input(shape)
    inner = residual_bone(input_layer, 3, 128, 1)
    inner = residual_bone(inner, 3, 128, 1)
    dense_layer = keras.layers.GlobalAveragePooling1D()(inner)
    dense_layer = keras.layers.Dense(units=32, activation='relu')(dense_layer)
    #dense_layer = keras.layers.Dropout(0.2)(dense_layer)
    output_layer = keras.layers.Dense(1, activation='sigmoid')(dense_layer) #softmax, sigmoid
    # output layer
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.AUC(name='auc'),
                           keras.metrics.Recall(name='Recall')
                         ])
    model.summary()
    return model
def transformer_encoder(shape):
    input_layer = keras.layers.Input(shape)
    conv1 = FCN_bone(input_layer, 48, 8)
    conv1 = FCN_bone(conv1, 64, 5)
    conv1 = FCN_bone(conv1, 120, 5)
    encoded_layer = get_encoders(
        encoder_num=8,
        input_layer=conv1,
        head_num=12,
        hidden_dim=24,
        attention_activation='relu',
        feed_forward_activation=gelu,
        dropout_rate= 0,
    )
    encoded_layer = keras.layers.Reshape((encoded_layer.shape[2],encoded_layer.shape[1]))(encoded_layer)
    gap_layer = keras.layers.GlobalAveragePooling1D()(encoded_layer)
    gap_layer = keras.layers.Dense(units=32, activation='relu')(gap_layer)

    output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer) #softmax, sigmoid
    # output layer
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.AUC(name='auc'),
                           keras.metrics.Recall(name='Recall')
                         ])
    model.summary()
    return model

GPU_num = 6
Type_num = 0
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
if gpus:
    tf.config.experimental.set_visible_devices(devices=(gpus[GPU_num]), device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[GPU_num], True)

def divide_train_val_by_kfold(k_fold, fold_sum, X_all,Y_all,Sample_info):
    num_val_samples = int(X_all.shape[0] // fold_sum)
    X_val = X_all[num_val_samples * k_fold:num_val_samples * (k_fold + 1)]
    X_train = np.concatenate((X_all[:num_val_samples * k_fold], X_all[num_val_samples * (k_fold + 1):]),axis =0)

    y_val = Y_all[num_val_samples * k_fold:num_val_samples * (k_fold + 1)]
    y_train = np.concatenate((Y_all[:num_val_samples * k_fold], Y_all[num_val_samples * (k_fold + 1):]),axis =0)

    sample_info_val = Sample_info.iloc[num_val_samples * k_fold:num_val_samples * (k_fold + 1)]
    return (X_train, y_train, X_val, y_val, sample_info_val)

#load your data
def load_dataset(name,random_seed):
    df_label = pd.read_csv(outputfolder + "one_wave_data/"+ name+"_all_id.csv", delimiter=",", index_col=0, header=0)
    all_label = df_label["label"].values
    Sample_info = df_label

    diease_data = np.load(outputfolder+ "one_wave_data/"+name+"_ecg_wave_460.npy", allow_pickle=True)
    diease_num = diease_data.shape[0]
    diease_lable = all_label[0:diease_num]
    noise_lable = all_label[diease_num:]

    print("diease_data shape:", diease_data.shape)
    noise_data = np.load(outputfolder+"one_wave_data/"+name+"_normal_wave_460.npy", allow_pickle=True)

    np.random.seed(random_seed)
    permutation = list(np.random.permutation(diease_data.shape[0]))
    diease = diease_data[permutation, :]
    diease_lable = diease_lable[permutation]
    permutation = list(np.random.permutation(noise_data.shape[0]))
    noise = noise_data[permutation, :]
    noise_lable = noise_lable[permutation]

    # 10% as test
    diease_test_size = diease.shape[0] // 10
    noise_test_size = noise.shape[0] // 10

    diease_test = diease[0:diease_test_size,:]
    noise_test = noise[0:noise_test_size, :]
    diease_val = diease[diease_test_size:, :]
    noise_val = noise[noise_test_size:, :]

    diease_lable_test = diease_lable[0:diease_test_size]
    noise_lable_test = noise_lable[0:noise_test_size]
    diease_lable_val = diease_lable[diease_test_size:]
    noise_lable_val = noise_lable[noise_test_size:]

    all_val = np.concatenate([diease_val, noise_val], axis=0)
    all_test = np.concatenate([diease_test, noise_test], axis=0)

    all_test_lable = np.concatenate([diease_lable_test, noise_lable_test], axis=0)
    all_val_lable = np.concatenate([diease_lable_val, noise_lable_val], axis=0)

    np.random.seed(2000)
    permutation = list(np.random.permutation(all_val.shape[0]))
    all_val = all_val[permutation, :]
    all_val_lable = all_val_lable[permutation]

    permutation = list(np.random.permutation(all_test.shape[0]))
    all_test = all_test[permutation, :, :]
    all_test_lable = all_test_lable[permutation]

    #all_val = all_val.swapaxes(1, 2)  #parallel
    all_val = tf.reshape(all_val, (all_val.shape[0], all_val.shape[2], all_val.shape[1])) # 12-lead Serial better
    all_test = tf.reshape(all_test, (all_test.shape[0], all_test.shape[2], all_test.shape[1]))

    Sample_info = Sample_info.iloc[permutation, :]
    return all_val, all_val_lable, all_test, all_test_lable,Sample_info



def run_model_in_k_fold(k_fold_sum, fold_index, Net_Id):

    (X_train, y_train, X_val, y_val, sample_info_val) = \
                      divide_train_val_by_kfold(fold_index, k_fold_sum, X_all, Y_all, Sample_info)
    results = []
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc', factor=0.1, patience=4, verbose=1, mode='max',
        min_delta=0.0001, cooldown=0, min_lr=0)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=15)

    if Net_Id == 0:
        model = FCN(X_train[0].shape)
    elif Net_Id == 1:
        model = Inception_cnn_1D(X_train[0].shape, feature_num[0])
    elif Net_Id == 2:
        model = Resnet_model(X_train[0].shape)
    elif Net_Id == 3 :
        model = LSTM_Model(X_train[0].shape, feature_num[0])
    elif Net_Id == 4:
        model = Attention_encoder_model(X_train[0].shape)
    elif Net_Id == 5:
        model = transformer_encoder(X_train[0].shape)
    elif Net_Id == 6:
        model = Dense(X_train[0].shape)

    model.fit(X_train, y_train, batch_size= batchsize, epochs = epochs_num,validation_data=(X_val,y_val),
                        callbacks=[reduce_lr,early_stop], verbose = 1)
    result = model.evaluate(X_val,y_val)
    results.append(result[2])
    y_val_pred = model.predict(X_val)
    results.append(result)

    model.fit(X_all,Y_all, batch_size= batchsize, epochs = 35, verbose = 1)
    result = model.evaluate(X_test,Y_test)
    results.append(result[2])
    y_test_pred = model.predict(X_test)
    results.append(result)

    return results


Net_name = ["CNN", "INCep", "RES", "LSTM", "Atten", "Transf", "Dense", "Transfer_Ptb"]
diease_name = "I251"
outputfolder = "your_data_path"

k_fold_sum = 5
ksize = [[8, 5, 3]]
feature_num = [64]
batchsize = 64
epochs_num = 60

random_seed = [2000]
for netid in range(2,3):
    for seed_id in range(0, len(random_seed)):
        (X_all,Y_all, X_test, Y_test, Sample_info) = load_dataset("I251", random_seed[seed_id] )
        Net_Id = netid
        k_fold_sum = 5
        for fold_index in range(0, k_fold_sum):
            keras.backend.clear_session()
            print(fold_index, k_fold_sum)
            results = run_model_in_k_fold(k_fold_sum, fold_index,Net_Id)


