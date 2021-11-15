#!/bin/env python

import sys
import os
from turtle import color

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Bio import SeqIO
import argparse
import psutil
import shutil, os
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import seaborn as sn
from joblib import dump

from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import tensorflow as tf

import time as tm
import datetime
import os
from operator import itemgetter
from numpy import argmax

# for working in Nvidia RTX 2080 super
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

"""
These functions are used to calculated perfomance metrics
"""
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def fasta2one_hot(sequence, total_win_len):
    langu = ['A', 'C', 'G', 'T', 'N']
    posNucl = 0
    if len(sequence) < total_win_len:
        rest = ['N' for x in range(total_win_len - len(sequence))]
        sequence += ''.join(rest)

    rep2d = np.zeros((1, 5, len(sequence)), dtype=np.int8)

    for nucl in sequence:
        posLang = langu.index(nucl.upper())
        rep2d[0][posLang][posNucl] = 1
        posNucl += 1
    return rep2d


def one_hot2fasta(dataset):
    langu = ['A', 'C', 'G', 'T', 'N']
    fasta_seqs = ""
    for j in range(dataset.shape[1]):
        if sum(dataset[:, j]) > 0:
            pos = argmax(dataset[:, j])
            fasta_seqs += langu[pos]
    return fasta_seqs


def metrics(Y_validation,predictions):
    classes = len(np.unique(Y_validation))
    print('Accuracy:', accuracy_score(Y_validation, predictions))
    print('F1 score:', f1_score(Y_validation, predictions,average='weighted'))
    print('Recall:', recall_score(Y_validation, predictions,average='weighted'))
    print('Precision:', precision_score(Y_validation, predictions, average='weighted'))
    print('\n clasification report:\n', classification_report(Y_validation, predictions))
    print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))
    #Creamos la matriz de confusión
    snn_cm = confusion_matrix(Y_validation, predictions)

    # Visualizamos la matriz de confusión
    snn_df_cm = pd.DataFrame(snn_cm, range(classes), range(classes))
    plt.figure(figsize = (20,14))
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 12}) # font size
    plt.show()


def graphics(history, AccTest, LossTest, log_Dir, model_Name, lossTEST, lossTRAIN, lossVALID, accuracyTEST,
             accuracyTRAIN, accuracyVALID):
    numbers = AccTest
    numbers_sort = sorted(enumerate(numbers), key=itemgetter(1), reverse=True)
    for i in range(int(len(numbers) * (0.05))):  # 5% Del total de las épocas
        index, value = numbers_sort[i]
        print("Test Accuracy {}, Época:{}\n".format(value, index + 1))

    print("")

    numbers = history.history['f1_m']
    numbers_sort = sorted(enumerate(numbers), key=itemgetter(1), reverse=True)
    for i in range(int(len(numbers) * (0.05))):  # 5% Del total de las épocas
        index, value = numbers_sort[i]
        print("Train Accuracy {}, Época:{}\n".format(value, index + 1))

    print("")

    numbers = history.history['val_f1_m']
    numbers_sort = sorted(enumerate(numbers), key=itemgetter(1), reverse=True)
    for i in range(int(len(numbers) * (0.05))):  # 5% Del total de las épocas
        index, value = numbers_sort[i]
        print("Validation F1-Score {}, Época:{}\n".format(value, index + 1))

    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(10, 10))
        # Plot training & validation accuracy values
        plt.plot(np.concatenate([np.array([accuracyTRAIN]), np.array(history.history['f1_m'])], axis=0))
        plt.plot(np.concatenate([np.array([accuracyVALID]), np.array(history.history['val_f1_m'])], axis=0))
        plt.plot(np.concatenate([np.array([accuracyTEST]), np.array(AccTest)], axis=0))  # Test
        plt.title('F1-Score Vs Epoch')
        plt.ylabel('F1-Score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
        plt.grid('on')
        # plt.savefig(path_img_base+'/Accuracy_GBRAS-Net_'+model_Name+'.eps', format='eps')
        # plt.savefig(path_img_base+'/Accuracy_GBRAS-Net_'+model_Name+'.svg', format='svg')
        # plt.savefig(path_img_base+'/Accuracy_GBRAS-Net_'+model_Name+'.pdf', format='pdf')
        # plt.show()

        plt.figure(figsize=(10, 10))
        # Plot training & validation loss values
        plt.plot(np.concatenate([np.array([lossTRAIN]), np.array(history.history['loss'])], axis=0))
        plt.plot(np.concatenate([np.array([lossVALID]), np.array(history.history['val_loss'])], axis=0))
        plt.plot(np.concatenate([np.array([lossTEST]), np.array(LossTest)], axis=0))  # Test
        plt.title('Loss Vs Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
        plt.grid('on')
        # plt.savefig(path_img_base+'/Loss_GBRAS-Net_'+model_Name+'.eps', format='eps')
        # plt.savefig(path_img_base+'/Loss_GBRAS-Net_'+model_Name+'.svg', format='svg')
        # plt.savefig(path_img_base+'/Loss_GBRAS-Net_'+model_Name+'.pdf', format='pdf')
        plt.show()


def Final_Results_Test(PATH_trained_models, X_test, Y_test):
    AccTest = []
    LossTest = []
    B_accuracy = 0  # B --> Best
    for filename in sorted(os.listdir(PATH_trained_models)):
        if filename != ('train') and filename != ('validation'):
            print(filename)
            model = tf.keras.models.load_model(PATH_trained_models + '/' + filename, custom_objects={'f1_m': f1_m})
            loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
            print(f'Loss={loss:.4f} y F1-Score={accuracy:0.4f}' + '\n')
            BandAccTest = accuracy
            BandLossTest = loss
            AccTest.append(BandAccTest)  # Valores de la precisión en Test, para graficar junto a valid y train
            LossTest.append(BandLossTest)  # Valores de la perdida en Test, para graficar junto a valid y train

            if accuracy > B_accuracy:
                B_accuracy = accuracy
                B_loss = loss
                B_name = filename

    print("\n\nBest")
    print(B_name)
    print(f'Loss={B_loss:.4f} y F1-Score={B_accuracy:0.4f}' + '\n')
    return AccTest, LossTest, B_name


def train(model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, epochs, path_log_base, model_name=""):
    start_time = tm.time()
    log_dir = path_log_base + "/" + model_name + "_" + str(
        datetime.datetime.now().isoformat()[:19].replace("T", "_").replace(":", "-"))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    filepath = log_dir + "/saved-model-{epoch:03d}-{val_f1_m:.4f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_f1_m', save_best_only=False, mode='max')
    model.reset_states()

    # VALORES EN TRAIN TEST Y VALIDACIÓN INICIALES, GRÁFICOS
    lossTEST, accuracyTEST = model.evaluate(X_test, y_test, verbose=None)
    lossVALID, accuracyVALID = model.evaluate(X_valid, y_valid, verbose=None)
    lossTRAIN, accuracyTRAIN = model.evaluate(X_train, y_train, verbose=None)

    model_Name = model_name
    log_Dir = log_dir

    history = model.fit(X_train, y_train, epochs=epochs,
                        callbacks=[tensorboard, checkpoint],
                        batch_size=batch_size, validation_data=(X_valid, y_valid), verbose=1)

    metrics = model.evaluate(X_test, y_test, verbose=0)

    TIME = tm.time() - start_time
    print("Time " + model_name + " = %s [seconds]" % TIME)

    print("\n")
    print(log_dir)
    return lossTEST, accuracyTEST, lossTRAIN, accuracyTRAIN, lossVALID, accuracyVALID, history

def kmer_extractor_model(dataset):
    # to load pre-calculated weights to extract k-mer frequencies
    installation_path = os.path.dirname(os.path.realpath(__file__))
    weights = np.load(installation_path + '/Models/Weights_SL.npy', allow_pickle=True)
    W_1 = weights[0]
    b_1 = weights[1]
    W_2 = weights[2]
    b_2 = weights[3]
    W_3 = weights[4]
    b_3 = weights[5]
    W_4 = weights[6]
    b_4 = weights[7]
    W_5 = weights[8]
    b_5 = weights[9]
    W_6 = weights[10]
    b_6 = weights[11]

    # to define the CNN model
    inputs = tf.keras.Input(shape=(dataset.shape[1], dataset.shape[2], 1), name="input_1")
    layers_1 = tf.keras.layers.Conv2D(4, (5, 1), strides=(1, 1), weights=[W_1, b_1], activation='relu',
                                      use_bias=True, name='k_1')(inputs)
    layers_1 = tf.keras.backend.sum(layers_1, axis=-2)

    layers_2 = tf.keras.layers.Conv2D(16, (5, 2), strides=(1, 1), weights=[W_2, b_2], activation='relu',
                                      use_bias=True, name='k_2')(inputs)
    layers_2 = tf.keras.backend.sum(layers_2, axis=-2)

    layers_3 = tf.keras.layers.Conv2D(64, (5, 3), strides=(1, 1), weights=[W_3, b_3], activation='relu',
                                      use_bias=True, name='k_3')(inputs)
    layers_3 = tf.keras.backend.sum(layers_3, axis=-2)

    layers_4 = tf.keras.layers.Conv2D(256, (5, 4), strides=(1, 1), weights=[W_4, b_4], activation='relu',
                                      use_bias=True, name='k_4')(inputs)
    layers_4 = tf.keras.backend.sum(layers_4, axis=-2)

    layers_5 = tf.keras.layers.Conv2D(1024, (5, 5), strides=(1, 1), weights=[W_5, b_5], activation='relu',
                                      use_bias=True, name='k_5')(inputs)
    layers_5 = tf.keras.backend.sum(layers_5, axis=-2)

    layers_6 = tf.keras.layers.Conv2D(4096, (5, 6), strides=(1, 1), weights=[W_6, b_6], activation='relu',
                                      use_bias=True, name='k_6')(inputs)
    layers_6 = tf.keras.backend.sum(layers_6, axis=-2)

    layers = tf.concat([layers_1, layers_2, layers_3, layers_4, layers_5, layers_6], 2)
    outputs = tf.keras.layers.Flatten()(layers)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    for layer in model.layers:
        layer.trainable = False
    return model


"""
This function calculates k-mer frequencies of seqFile and write them in the seqfile.kmers file
"""
def k_mer_counting(seqFile, outputDir, total_win_len, lineage_names):
    kmer_extractor = kmer_extractor_model(np.zeros((1, 5, total_win_len)))
    file_name = os.path.basename(seqFile)
    result_file = open(outputDir+'/'+file_name+'.kmers', 'w')
    seqs = SeqIO.parse(seqFile, "fasta")

    # to put the headers
    kmers = []
    for k in range(1, 7):
        for item in itertools.product('ACGT', repeat=k):
            kmers.append(''.join(item))
    if lineage_names.upper() == 'YES':
        result_file.write('Label,' + ','.join(kmers) + '\n')
    else:
        result_file.write(','.join(kmers) + '\n')

    for seq in seqs:
        TEid = str(seq.id)
        if lineage_names.upper() == 'YES':
            order = -1
            if str(TEid).upper().find("ALE-") != -1 or str(TEid).upper().find("RETROFIT-") != -1:
                order = 1
            elif str(TEid).upper().find("ALESIA-") != -1:
                order = 2
            elif str(TEid).upper().find("ANGELA-") != -1:
                order = 3
            elif str(TEid).upper().find("BIANCA-") != -1:
                order = 4
            elif str(TEid).upper().find("BRYCO-") != -1:
                order = 5
            elif str(TEid).upper().find("LYCO-") != -1:
                order = 6
            elif str(TEid).upper().find("GYMCO-") != -1:
                order = 7
            elif str(TEid).upper().find("IKEROS-") != -1:
                order = 8
            elif str(TEid).upper().find("IVANA-") != -1 or str(TEid).upper().find("ORYCO-") != -1:
                order = 9
            elif str(TEid).upper().find("OSSER-") != -1:
                order = 10
            elif str(TEid).upper().find("TAR-") != -1:
                order = 11
            elif str(TEid).upper().find("TORK-") != -1:
                order = 12
            elif str(TEid).upper().find("SIRE-") != -1:
                order = 13
            elif str(TEid).upper().find("CRM-") != -1:
                order = 14
            elif str(TEid).upper().find("CHLAMYVIR-") != -1:
                order = 15
            elif str(TEid).upper().find("GALADRIEL-") != -1:
                order = 16
            elif str(TEid).upper().find("REINA-") != -1:
                order = 17
            elif str(TEid).upper().find("TEKAY-") != -1 or str(TEid).upper().find("DEL-") != -1:
                order = 18
            elif str(TEid).upper().find("ATHILA-") != -1:
                order = 19
            elif str(TEid).upper().find("TAT-") != -1:
                order = 20
            elif str(TEid).upper().find("OGRE-") != -1:
                order = 21
            elif str(TEid).upper().find("RETAND-") != -1:
                order = 22
            elif str(TEid).upper().find("PHYGY-") != -1:
                order = 23
            elif str(TEid).upper().find("SELGY-") != -1:
                order = 24

            if order != -1:
                kmer_counts = kmer_extractor.predict(fasta2one_hot(str(seq.seq), total_win_len))
                result_file.write(str(order)+','+','.join([str(int(kmer_counts[0, f])) for f in range(kmer_counts.shape[1])])+'\n')
        else:
            kmer_counts = kmer_extractor.predict(fasta2one_hot(str(seq.seq), total_win_len))
            result_file.write(','.join([str(int(kmer_counts[0, f])) for f in range(kmer_counts.shape[1])]) + '\n')

    result_file.close()

def Inpactor2_Class(X_train):
    tf.keras.backend.clear_session()

    #Inputs
    inputs = tf.keras.Input(shape=(X_train.shape[1],), name="input_1")
    #layer 1
    layers = tf.keras.layers.Dense(200,activation="relu", kernel_regularizer=regularizers.l1(0.0001),bias_regularizer=regularizers.l2(0.01))(inputs)
    layers = tf.keras.layers.Dropout(0.5)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #layer 2
    layers = tf.keras.layers.Dense(200,activation="relu", kernel_regularizer=regularizers.l1(0.0001),bias_regularizer=regularizers.l2(0.01))(layers)
    layers = tf.keras.layers.Dropout(0.5)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #layer 3
    layers = tf.keras.layers.Dense(200,activation="relu", kernel_regularizer=regularizers.l1(0.0001),bias_regularizer=regularizers.l2(0.01))(layers)
    layers = tf.keras.layers.Dropout(0.5)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    # layer 4
    predictions = tf.keras.layers.Dense(21, activation="softmax", name="output_1")(layers)
    # model generation
    model = tf.keras.Model(inputs = inputs, outputs=predictions)
    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,)
    # loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    # Compile model
    model.compile(loss=loss_fn, optimizer=opt, metrics=[f1_m])
    return model

def retraining_class(kmer_file, outputDir):

    #load k-mer file and separate features from labels
    training_data = pd.read_csv(kmer_file)
    label_vectors = training_data['Label'].values
    feature_vectors = training_data.drop(['Label'], axis=1).values
    model_name = "Inpactor2_Class"
    log_dir = outputDir+'/logs' + "/" + model_name + "_" + str(
        datetime.datetime.now().isoformat()[:19].replace("T", "_").replace(":", "-"))

    # Scaling
    scaler = preprocessing.StandardScaler().fit(feature_vectors)
    feature_vectors_scaler = scaler.transform(feature_vectors)

    #data split: 80% train, 10% dev and 10% test
    validation_size = 0.2
    seed = 7
    X_trainScaler, X_test_dev, Y_trainScaler, Y_test_dev = train_test_split(feature_vectors_scaler, label_vectors,
                                                                                            test_size=validation_size,
                                                                                            random_state=seed)
    X_dev, X_test, Y_dev, Y_test = train_test_split(X_test_dev, Y_test_dev, test_size=0.5, random_state=seed)
    feature_vectors = None
    label_vectors = None

    dump(scaler, outputDir+'/std_scaler.bin', compress=True)

    # PCA dimentional reduction
    pca = decomposition.PCA(n_components=0.96, svd_solver='full', tol=1e-4)
    pca.fit(X_trainScaler)
    X_trainPCAScaler = pca.transform(X_trainScaler)
    X_validationPCAScaler = pca.transform(X_dev)
    X_testPCAScaler = pca.transform(X_test)
    dump(pca, outputDir+'/std_pca.bin', compress=True)

    # to train the DNN architecture
    model = Inpactor2_Class(X_trainPCAScaler)
    # summarize layers
    print(model.summary())
    tf.keras.utils.plot_model(model, show_shapes=True)

    one_hot_labels_train = tf.keras.utils.to_categorical(Y_trainScaler, num_classes=21)
    one_hot_labels_validation = tf.keras.utils.to_categorical(Y_dev, num_classes=21)
    one_hot_labels_test = tf.keras.utils.to_categorical(Y_test, num_classes=21)

    # Fit the model
    lossTEST, accuracyTEST, lossTRAIN, accuracyTRAIN, lossVALID, accuracyVALID, history= train(model, X_trainPCAScaler,
          one_hot_labels_train, X_validationPCAScaler, one_hot_labels_validation, X_testPCAScaler, one_hot_labels_test,
                                                                                         128, 200, log_dir, model_name)
    AccTest, LossTest, B_name = Final_Results_Test(log_dir, X_test, one_hot_labels_test)

    # plot metrics
    plt.plot(history.history['f1_m'])
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('Epoch vs F1-Score')
    plt.show()

    # GRÁFICOS DE LAS TRES CURVAS TRAIN TEST Y VALIDACIÓN
    graphics(history, AccTest, LossTest, log_dir, model_name, lossTEST, lossTRAIN, lossVALID, accuracyTEST,
             accuracyTRAIN, accuracyVALID)

    # to test the perfomance
    model = tf.keras.models.load_model(log_dir + '/'+B_name, custom_objects={'f1_m': f1_m})

    scores = model.evaluate(X_trainPCAScaler, one_hot_labels_train, verbose=0)
    print("Baseline Error train: %.2f%%" % (100 - scores[1] * 100))

    scores = model.evaluate(X_validationPCAScaler, one_hot_labels_validation, verbose=0)
    print("Baseline Error dev: %.2f%%" % (100 - scores[1] * 100))

    scores = model.evaluate(X_testPCAScaler, one_hot_labels_test, verbose=0)
    print("Baseline Error test: %.2f%%" % (100 - scores[1] * 100))

    predictions = model.predict(X_trainPCAScaler)
    metrics(Y_trainScaler, [argmax(x) for x in predictions])
    predictions = model.predict(X_validationPCAScaler)
    metrics(Y_dev, [argmax(x) for x in predictions])
    predictions = model.predict(X_testPCAScaler)
    metrics(Y_test, [argmax(x) for x in predictions])

    shutil.move(log_dir + '/'+B_name, outputDir+'/Inpactor_Class.hdf5')
    os.remove(log_dir)

"""
This function deletes all characters that are no DNA (A, C, G, T, N)
"""
def filter(file, outputDir):
    basename = os.path.basename(file)
    newFile = open(outputDir+"/"+basename+".filtered", "w")
    for te in SeqIO.parse(file, "fasta"):
        seq = str(te.seq)
        filterDna = [x for x in seq if x.upper() in ['A', 'C', 'G', 'T', 'N']]
        newSeq = "".join(filterDna)
        newFile.write(">"+str(te.id)+"\n"+newSeq+"\n")

if __name__ == '__main__':
    print("\n#############################################")
    print("#                                           #")
    print("# Inpactor2 Utils: Utilities for Inpactor2  #")
    print("#                                           #")
    print("#############################################\n")

    ### read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--util', required=True, dest='util', help='Utility to be used [FILTER, CLASSIFY, KMER]')
    parser.add_argument('-o', '--output-dir', required=True, dest='outputDir', help='Path of the output directory')
    parser.add_argument('-t', '--threads', required=False, dest='threads',
                        help='Number of threads to be used by Inpactor2')
    parser.add_argument('-f', '--fasta-file', required=False, dest='fastafile', help='Path of fasta file containg DNA sequences (for KMER and CLASSIFY utils)')
    parser.add_argument('-l', '--lineage-names', required=False, dest='lineage_names',
                        help='fasta file includes lineage names? [yes or not] (for KMER util)')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v1.0')

    options = parser.parse_args()
    util = options.util
    outputDir = options.outputDir
    threads = options.threads
    fastafile = options.fastafile
    lineage_names = options.lineage_names

    ##################################################################################
    # global configuration variables
    total_win_len = 50000

    ##############################################################################
    # Parameters' validation

    if util is None:
        print('FATAL ERROR: Missing utility parameter (-u or --util). Exiting')
        sys.exit(0)
    elif util.upper() not in ['DETECT', 'FILTER', 'CLASSIFY', 'KMER']:
        print('FATAL ERROR: '+util+' not found, utility must be one of the following: DETECT, FILTER, CLASSIFY, KMER')
        sys.exit(0)
    if outputDir is None:
        print('FATAL ERROR: Missing output directory parameter (-o or --output-dir). Exiting')
        sys.exit(0)
    elif not os.path.exists(outputDir):
        print('FATAL ERROR: output directory did not found at path: ' + outputDir)
        sys.exit(0)
    if threads is None or threads == -1:
        threads = int(psutil.cpu_count())
        print("WARNING: Missing threads parameter, using by default: " + str(threads))
    else:
        threads = int(threads)

    ##################################################################################
    # First Util: to count k-mer frequencies (1 <= k <= 6)
    if util.upper() == "KMER":
        if fastafile is None:
            print('FATAL ERROR: Missing fasta file parameter (-f or --fasta-file). Existing')
            sys.exit(0)
        elif not os.path.exists(fastafile):
            print('FATAL ERROR: Fasta file did not found at path: ' + fastafile)
            sys.exit(0)
        if lineage_names is None:
            print("WARNING: Missing -l or --lineage-names parameter, using by default: yes")
        elif lineage_names.upper() not in ['YES', 'NO']:
            print('FATAL ERROR: Incorrect value for -l or --lineage-names parameter: '+lineage_names+'. Must be yes or not. Existing')
            sys.exit(0)
        k_mer_counting(fastafile, outputDir, total_win_len, lineage_names)

    ##################################################################################
    # Second Util: re-training Inpactor2_Class
    if util.upper() == "CLASSIFY":
        if fastafile is None:
            print('FATAL ERROR: Missing fasta file parameter (-f or --fasta-file). Existing')
            sys.exit(0)
        elif not os.path.exists(fastafile):
            print('FATAL ERROR: Fasta file did not found at path: ' + fastafile)
            sys.exit(0)
        k_mer_counting(fastafile, outputDir, total_win_len)

        # To call the deep neural network
        retraining_class(outputDir+'/'+fastafile+'.kmers', outputDir)

    ##################################################################################
    # Third Util: filtering characters that are not nucleotides (A, C, G, T or N)
    if util.upper() == "FILTER":
        if fastafile is None:
            print('FATAL ERROR: Missing fasta file parameter (-f or --fasta-file). Existing')
            sys.exit(0)
        elif not os.path.exists(fastafile):
            print('FATAL ERROR: Fasta file did not found at path: ' + fastafile)
            sys.exit(0)
        filter(fastafile, outputDir)
