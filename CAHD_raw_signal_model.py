import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from keras_transformer import get_encoders
from keras_transformer import gelu
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential,load_model

os.environ['TF_KERAS']='1'
os.putenv('TF_KERAS', '1')
os.environ.setdefault('TF_KERAS', '1')

def FCN_bone(input_layber, features, ksize):
    conv1 = keras.layers.Conv1D(filters=features, kernel_size=ksize, padding='same')(input_layber)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)
    conv1 = keras.layers.MaxPool1D()(conv1)

    return  conv1
def FCN(shape):
    inputlayer = keras.layers.Input(shape)

    conv1 = FCN_bone(inputlayer, 48, 8)
    conv1 = FCN_bone(conv1, 64, 5)
    conv1 = FCN_bone(conv1, 128, 5)
    conv1 = FCN_bone(conv1, 256, 3)
    conv1 = FCN_bone(conv1, 384, 3)
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
    cnn_concat = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=(5), padding='same')(incep_bone)
    cnn_concat = keras.layers.BatchNormalization()(cnn_concat)
    cnn_concat = keras.layers.Activation('relu')(cnn_concat)
    cnn_concat = keras.layers.MaxPool1D()(cnn_concat)

    incep_bone = Inception_bone(cnn_concat, n_feature_maps*4)
    cnn_concat = keras.layers.Conv1D(filters=n_feature_maps * 4, kernel_size=(5), padding='same')(incep_bone)
    cnn_concat = keras.layers.BatchNormalization()(cnn_concat)
    cnn_concat = keras.layers.Activation('relu')(cnn_concat)
    cnn_concat = keras.layers.MaxPool1D()(cnn_concat)

    incep_bone = Inception_bone(cnn_concat, n_feature_maps * 8)
    cnn_concat = keras.layers.Conv1D(filters=n_feature_maps*8, kernel_size=(5), padding='same')(incep_bone)
    cnn_concat = keras.layers.BatchNormalization()(cnn_concat)
    cnn_concat = keras.layers.Activation('relu')(cnn_concat)
    cnn_concat = keras.layers.MaxPool1D()(cnn_concat)

    gap_layer = keras.layers.GlobalAveragePooling1D()(cnn_concat)
    gap_layer = keras.layers.Dense(units=32, activation='relu')(gap_layer)
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
    conv1 = FCN_bone(input_layer, 48, 8)
    conv1 = FCN_bone(conv1, 64, 5)
    # conv block -1
    conv3 = keras.layers.Conv1D(filters=512, kernel_size=5, padding='same')(conv1)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
    conv3 = keras.layers.MaxPooling1D()(conv3)
    # split for attention
    attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(conv3)
    attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(conv3)
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

    cnn_layer = FCN_bone(inputs,48, 8)
    cnn_layer = FCN_bone(cnn_layer, 64,5)
    cnn_layer = FCN_bone(cnn_layer, 96,3)
    # CNN to RNN
    lstm_1 = keras.layers.LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(
        cnn_layer)  # (None, 32, 512)
    # lstm_1b = keras.layers.LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(
    #     cnn_layer)
    # reversed_lstm_1b = keras.layers.Lambda(lambda inputTensor: keras.backend.reverse(inputTensor, axes=1))(lstm_1b)
    # lstm1_merged = keras.layers.add([lstm_1, reversed_lstm_1b])  # (None, 32, 512)
    # lstm1_merged = keras.layers.BatchNormalization()(lstm1_merged)

    lstm_2 = keras.layers.LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm_1)#lstm1_merged
    # lstm_2b = keras.layers.LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
    #     lstm1_merged)
    # reversed_lstm_2b = keras.layers.Lambda(lambda inputTensor: keras.backend.reverse(inputTensor, axes=1))(lstm_2b)
    # lstm2_merged = keras.layers.concatenate([lstm_2, reversed_lstm_2b])  # (None, 32, 1024)concatenate
    # lstm2_merged = keras.layers.BatchNormalization()(lstm2_merged)
    # transforms RNN output to character activations:
    gap_layer = keras.layers.GlobalAveragePooling1D()(lstm_2)
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
def residual_bone(input_layer,ksize, feature_num,is_pool):
    n_feature_maps = feature_num
    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, padding='same', kernel_size=ksize[0])(input_layer)#
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps, padding='same', kernel_size=ksize[1])(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps, padding='same', kernel_size=ksize[2])(conv_y)
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

def Resnet_model(shape, ksize, feature_num):
    input_layer = keras.layers.Input(shape)

    conv1 = FCN_bone(input_layer, 48, 5)
    conv1 = FCN_bone(conv1, 64, 3)
    inner = residual_bone(conv1, ksize, feature_num, 0)
    inner = residual_bone(inner, ksize, feature_num * 2, 0)
    inner = residual_bone(inner, ksize, feature_num * 4, 0)
    inner = residual_bone(inner, ksize, feature_num * 8, 0)

    dense_layer = keras.layers.GlobalAveragePooling1D()(inner)
    dense_layer = keras.layers.Dense(units=32, activation='relu')(dense_layer)
    #dense_layer = keras.layers.Dense(units=8, activation='relu')(dense_layer)
    #dense_layer = keras.layers.Dropout(0.1)(dense_layer)
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
    conv1 = FCN_bone(conv1, 48, 5)
    conv1 = FCN_bone(conv1, 64, 3)
    conv1 = FCN_bone(conv1, 64, 3)
    encoded_layer = get_encoders(
        encoder_num=6,
        input_layer=conv1,
        head_num=8,
        hidden_dim=24,
        attention_activation='relu',
        feed_forward_activation=gelu,
        dropout_rate= 0.1,
    )

    encoded_layer = keras.layers.Reshape((encoded_layer.shape[2],encoded_layer.shape[1]))(encoded_layer)
    gap_layer = keras.layers.GlobalAveragePooling1D()(encoded_layer)
    gap_layer = keras.layers.Dense(units=32, activation='relu')(gap_layer)
    #gap_layer = keras.layers.Dropout(0)(gap_layer)

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

def Transfer_learn_resnet(ptb_type_name):
    model = load_model(model_path + "PTB_"+ptb_type_name+".h5")
    ptb_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('global_average_pooling1d').output) # 'dense' #'global_average_pooling1d'
    new_model = Sequential()
    new_model.add(ptb_model)
    new_model.add(keras.layers.Dense(32, activation='relu'))
    new_model.add(keras.layers.Dense(1, activation='sigmoid'))
    new_model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                           keras.metrics.AUC(name='auc')
                         ])
    new_model.summary()

    return  new_model

GPU_num = 3#int(sys.argv[1])
Type_num = 0#int(sys.argv[2])
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
if gpus:
    tf.config.experimental.set_visible_devices(devices=(gpus[GPU_num]), device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[GPU_num], True)
diease_name = "I251" # CAHD code
outputfolder = "your_path"
model_path = "your_Model_save_path"

def divide_train_val_by_kfold(k_fold, fold_sum, X_all,Y_all,Sex_age_Label,Sample_info):
    num_val_samples = int(X_all.shape[0] // fold_sum)
    X_val = X_all[num_val_samples * k_fold:num_val_samples * (k_fold + 1)]
    X_train = np.concatenate((X_all[:num_val_samples * k_fold], X_all[num_val_samples * (k_fold + 1):]),axis =0)

    y_val = Y_all[num_val_samples * k_fold:num_val_samples * (k_fold + 1)]
    y_train = np.concatenate((Y_all[:num_val_samples * k_fold], Y_all[num_val_samples * (k_fold + 1):]),axis =0)

    sex_val = Sex_age_Label[num_val_samples * k_fold:num_val_samples * (k_fold + 1)]
    sex_train = np.concatenate((Sex_age_Label[:num_val_samples * k_fold], Sex_age_Label[num_val_samples * (k_fold + 1):]),axis =0)

    sample_info_val = Sample_info.iloc[num_val_samples * k_fold:num_val_samples * (k_fold + 1)]

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, sex_train.shape, sex_val.shape)
    return (X_train, y_train, X_val, y_val, sex_train, sex_val, sample_info_val)

def load_dataset(name):
    df_label = pd.read_csv(outputfolder + name+"_and_noise_id.csv", delimiter=",", index_col=0, header=0)
    all_label = df_label["label"].values
    sex_age = df_label.iloc[:, [0, 1]]
    Sample_info = df_label

    sex_age_arr = sex_age.values
    mean = sex_age_arr.mean(axis=0)
    sex_age_arr -= mean
    std = sex_age_arr.std(axis=0)
    sex_age_arr /= std

    diease_data = np.load(outputfolder+name+"_ecg_data.npy", allow_pickle=True)
    noise_data = np.load(outputfolder+name+"_noise_ecg_data.npy", allow_pickle=True)

    X_all = np.concatenate((diease_data, noise_data), axis=0)
    Y_all = all_label
    # mean = X_all.mean(axis=0)
    # X_all -= mean
    # std = X_all.std(axis=0)
    # X_all /= std
    #X_all = X_all.swapaxes(1, 2)  #parallel
    X_all = tf.reshape(X_all, (X_all.shape[0], X_all.shape[2], X_all.shape[1])) # 12-lead Serial
    X_all = np.array(X_all)

    np.random.seed(1000)
    permutation = list(np.random.permutation(X_all.shape[0]))
    X_shuf = X_all[permutation, :, :]
    Y_shuf = Y_all[permutation]
    Sex_age_Label = sex_age_arr[permutation, :]
    Sample_info = Sample_info.iloc[permutation, :]

    return X_shuf,Y_shuf,Sex_age_Label,Sample_info

def Save_CV_predict_result(sample_info_val, y_pred, Net_id):
    sample_info_val["PredictLabel"] = y_pred
    if (os.path.exists(outputfolder+"CV_result/"+Net_name[Net_id]+ "_cv.csv")):
        X_exist =  pd.read_csv(outputfolder +"CV_result/"+Net_name[Net_id]+ "_cv.csv", delimiter=",", index_col = 0,header = 0)
        X_exist = pd.concat((X_exist, sample_info_val))
        X_exist.to_csv(outputfolder +"CV_result/"+Net_name[Net_id]+ "_cv.csv")
    else:
        sample_info_val.to_csv(outputfolder+"CV_result/"+Net_name[Net_id]+ "_cv.csv")

def run_model_in_k_fold(k_fold_sum, fold_index, Net_Id):
    ksize = [[5, 3, 3]]  # ,[3,3,3][8,5,3],,[17, 11, 9, 5, 3]
    feature_num = [64]  # 64,
    epochs_num = 50
    (X_train, y_train, X_val, y_val, sex_train, sex_val, sample_info_val) = \
                      divide_train_val_by_kfold(fold_index, k_fold_sum, X_all, Y_all, Sex_age_Label,Sample_info)
    results = []
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc', factor=0.1, patience=4, verbose=1, mode='max',
        min_delta=0.0001, cooldown=0, min_lr=0)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=7)

    if Net_Id == 0:
        model = FCN(X_train[0].shape)
    elif Net_Id == 1:
        model = Inception_cnn_1D(X_train[0].shape, feature_num[0])
    elif Net_Id == 2:
        model = Resnet_model(X_train[0].shape, ksize[0], feature_num[0])
    elif Net_Id == 3 :
        model = LSTM_Model(X_train[0].shape, feature_num[0])
    elif Net_Id == 4:
        model = Attention_encoder_model(X_train[0].shape)
    elif Net_Id == 5:
        model = transformer_encoder(X_train[0].shape)
    model.fit(X_train, y_train, batch_size= batchsize, epochs = epochs_num,validation_data=(X_val,y_val),
                        callbacks=[reduce_lr, early_stop], verbose = 1)  # callbacks=[reduce_lr, early_stop],validation_split = 0.2,

    model.save(outputfolder +"/Model/"+ Net_name[Net_Id]+"_"+str(fold_index)+"_cv.h5")
    result = model.evaluate(X_val,y_val)
    results.append(result[2])
    y_val_pred = model.predict(X_val)

    result = f1_score(y_val,y_val_pred.round(),average='micro')
    results.append(result)
    Save_CV_predict_result(sample_info_val, y_val_pred, Net_Id)

    return results


Net_name = ["CNN", "INCep", "RES", "LSTM", "Atten", "Transf"]
batchsize = 96
for netid in range(0,6):
    (X_all,Y_all,Sex_age_Label, Sample_info) = load_dataset(diease_name)
    print("X_all shape:", X_all.shape)
    k_fold_sum = 5
    for fold_index in range(0,k_fold_sum):
        keras.backend.clear_session()
        print(fold_index, k_fold_sum)
        results = run_model_in_k_fold(k_fold_sum, fold_index,netid)
        file_proce = open(outputfolder+"output/"+Net_name[netid]+ ".txt", "a")
        pro_loss = str("k_flod_num: " +str(fold_index) + ",AUC_F1:" + str(results) + "\n")
        file_proce.write(pro_loss)
        file_proce.close()
