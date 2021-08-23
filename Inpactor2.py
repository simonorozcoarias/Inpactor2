#!/bin/env python

import sys
import os
from turtle import color

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from Bio import SeqIO
import subprocess
import time
import multiprocessing
import argparse
import psutil
from joblib import dump, load
import tensorflow as tf
from tensorflow.keras import backend as K
from numpy import argmax
import numpy as np


# Uncomment the following lines for working in Nvidia RTX 2080 super
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

"""
These functions are used to calculated performance metrics
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

def check_nucleotides_master(list_seqs, threads):
    n = len(list_seqs)
    seqs_per_procs = int(n / threads)
    remain = n % threads
    ini_per_thread = []
    end_per_thread = []
    for p in range(threads):
        if p < remain:
            init = p * (seqs_per_procs + 1)
            end = n if init + seqs_per_procs + 1 > n else init + seqs_per_procs + 1
        else:
            init = p * seqs_per_procs + remain
            end = n if init + seqs_per_procs > n else init + seqs_per_procs
        ini_per_thread.append(init)
        end_per_thread.append(end)

    # Run in parallel the checking
    pool = multiprocessing.Pool(processes=threads)
    localresults = [pool.apply_async(check_nucleotides_slave,
                                     args=[list_seqs[ini_per_thread[x]:end_per_thread[x]]]) for x in range(threads)]
    localChecks = [p.get() for p in localresults]
    for i in range(len(localChecks)):
        if localChecks[i] == 1:
            print("FATAL ERROR: DNA sequences must contain only A, C, G, T, or N characters, please fix it and "
                  "re-run Inpactor2")
            sys.exit(0)

    pool.close()

def check_nucleotides_slave(list_seqs):
    for seq in list_seqs:
        noDNAlanguage = [nucl for nucl in str(seq) if nucl.upper() not in ['A', 'C', 'T', 'G', 'N', '\n']]
        if len(noDNAlanguage) > 0:
            return 1
    return 0

def create_dataset_master(list_ids, list_seqs, threads, total_win_len, outputDir):
    n = len(list_ids)
    seqs_per_procs = int(n / threads)
    remain = n % threads
    ini_per_thread = []
    end_per_thread = []
    for p in range(threads):
        if p < remain:
            init = p * (seqs_per_procs + 1)
            end = n if init + seqs_per_procs + 1 > n else init + seqs_per_procs + 1
        else:
            init = p * seqs_per_procs + remain
            end = n if init + seqs_per_procs > n else init + seqs_per_procs
        ini_per_thread.append(init)
        end_per_thread.append(end)
    pool = multiprocessing.Pool(processes=threads)

    localresults = [pool.apply_async(create_dataset_slave,
                                     args=[list_seqs[ini_per_thread[x]:end_per_thread[x]], total_win_len, outputDir,
                                           x]) for x in
                    range(threads)]
    localTables = [p.get() for p in localresults]

    splitted_genome = np.zeros((n, 5, total_win_len), dtype=bool)
    index = 0
    for i in range(len(localTables)):
        if localTables[i].shape[0] > 1:
            try:
                dataset = np.load(outputDir + '/dataset_2d_' + str(i) + '.npy')
                for j in range(dataset.shape[0]):
                    splitted_genome[index, :, :] = dataset[j, :, :]
                    index += 1
                os.remove(outputDir + '/dataset_2d_' + str(i) + '.npy')
            except FileNotFoundError:
                print('WARNING: I could not find: ' + outputDir + '/dataset_2d_' + str(i) + '.npy')
    pool.close()
    return splitted_genome

def create_dataset_slave(list_seqs, total_win_len, outputdir, x):
    j = 0
    if len(list_seqs) > 0:
        dataset = np.zeros((len(list_seqs), 5, total_win_len), dtype=bool)
        for i in range(len(list_seqs)):
            dataset[j, :, :] = fasta2one_hot(list_seqs[i], total_win_len)
            j += 1

        if dataset.shape[1] > 1:
            np.save(outputdir + '/dataset_2d_' + str(x) + '.npy', dataset.astype(np.uint8))
            return np.zeros((10, 10), dtype=bool)
        else:  # Process did not find any LTR-RT
            return np.zeros((1, 1), dtype=bool)
    else:
        # there is no elements for processing in this thread
        return np.zeros((1, 1), dtype=bool)


def get_final_dataset_size(file, total_win_len, slide):
    seqfile = [x for x in SeqIO.parse(file, 'fasta')]
    list_ids_splitted = []
    list_seq_splitter = []
    for i in range(len(seqfile)):
        for j in range(slide, len(str(seqfile[i].seq)), total_win_len):
            if "#" in str(seqfile[i].id):
                print("FATAL ERROR: Sequence ID (" + str(seqfile[i].id) + ") must no contain character '#', please remove "
                                     "all of these and re-run Inpactor2")
                sys.exit(0)
            initial_pos = j
            end_pos = initial_pos + total_win_len
            if end_pos > len(str(seqfile[i].seq)):
                end_pos = len(str(seqfile[i].seq))
            list_ids_splitted.append(str(seqfile[i].id) + "#" + str(initial_pos) + "#" + str(end_pos))
            list_seq_splitter.append(str(seqfile[i].seq)[initial_pos:end_pos])
    return list_ids_splitted, list_seq_splitter


def fasta2one_hot(sequence, total_win_len):
    langu = ['A', 'C', 'G', 'T', 'N']
    posNucl = 0
    rep2d = np.zeros((1, 5, total_win_len), dtype=bool)

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


"""
This function predicts which windows contains LTR-retrotransposons
"""
def Inpactor2_Detect(splitted_genome, detec_threshold, list_ids):
    new_list_ids = []
    installation_path = os.path.dirname(os.path.realpath(__file__))
    model = tf.keras.models.load_model(installation_path + '/Models/Inpactor_Detect_model.hdf5')

    predictions = model.predict(splitted_genome)
    predicted_windows = len([x for x in range(predictions.shape[0]) if predictions[x, 0] > detec_threshold])
    splitted_genome_ltr = np.zeros((predicted_windows, splitted_genome.shape[1], splitted_genome.shape[2]), dtype=bool)
    detect_proba = []

    j = 0  # index of the newly spitted_genome_ltr array
    for i in range(predictions.shape[0]):
        if predictions[i, 0] > detec_threshold:
            splitted_genome_ltr[j, :, :] = splitted_genome[i, :, :]
            detect_proba.append(predictions[i, 0])
            new_list_ids.append(list_ids[i])
            j += 1

    return splitted_genome_ltr, detect_proba, new_list_ids


def sequences_extractor_master(splitted_genome_ltr, threads, outputDir, max_len_threshold, min_len_threshold, list_ids, tg_ca, TSD, detection_proba):
    n = splitted_genome_ltr.shape[0]
    seqs_per_procs = int(n / threads)
    remain = n % threads
    splitted_genome_list = []

    for i in range(threads):
        if i < remain:
            init = i * (seqs_per_procs + 1)
            end = init + seqs_per_procs + 1
        else:
            init = i * seqs_per_procs + remain
            end = n if init + seqs_per_procs > n else init + seqs_per_procs
        splitted_genome_list.append(splitted_genome_ltr[init:end, :, :])

    pool = multiprocessing.Pool(processes=threads)
    localresults = [pool.apply_async(sequences_extractor_slave,
                                     args=[splitted_genome_list[x], x, seqs_per_procs, n, remain, outputDir,
                                           max_len_threshold, min_len_threshold, list_ids, tg_ca, TSD,
                                           detection_proba]) for x in range(threads)]
    localTables = [p.get() for p in localresults]

    # to join local results of extracted sequences
    pos_predicted = []
    for i in range(0, len(localTables)):
        pos_predicted.extend(localTables[i])

    # to join local results of predicted IDs
    ids_predicted = []
    for i in range(threads):
        try:
            IDsfile = open(outputDir + '/predicted_ids_' + str(i) + '.txt', 'r')
            lines = IDsfile.readlines()
            for line in lines:
                ids_predicted.append(line.replace('\n', ''))
            IDsfile.close()
            os.remove(outputDir + '/predicted_ids_' + str(i) + '.txt')
        except FileNotFoundError:
            print('WARNING: I could not find: ' + outputDir + '/predicted_ids_' + str(i) + '.txt')
    pool.close()

    return pos_predicted, ids_predicted


def sequences_extractor_slave(splitted_genome, x, seqs_per_procs, n, remain, outputdir,
                        max_len_threshold, min_len_threshold, list_ids, tg_ca, TSD, detect_proba):
    if x < remain:
        i = x * (seqs_per_procs + 1)
        m = n if i + seqs_per_procs + 1 > n else i + seqs_per_procs + 1
    else:
        i = x * seqs_per_procs + remain
        m = n if i + seqs_per_procs > n else i + seqs_per_procs

    if i < m:
        predicted_ids = []
        predicted_pos = []

        k = 0  # index of the splitted_genome dataset of this thread
        while i < m and k < len(splitted_genome):
            #######
            # to get the positions of the element in the window.
            bestCandidates = adjust_seq_positions(splitted_genome[k, :, :], outputdir, x, max_len_threshold,
                                                  min_len_threshold, tg_ca, TSD)
            if len(bestCandidates) > 0:
                j = 0  # index of the extracted_seq_i np array
                for c in range(len(bestCandidates)):
                    init = bestCandidates[c][0]
                    end = bestCandidates[c][1]
                    j += 1
                    # to extract the seq ID to save in the new predicted_list
                    seq_id = list_ids[i].split("#")[0]
                    factor = int(list_ids[i].split("#")[1])
                    predicted_ids.append(
                        seq_id + "#" + str(init + factor) + "#" + str(end + factor) + "#" + str(detect_proba[i]))
                    predicted_pos.append([init, end, i])

            i += 1
            k += 1

        IDsfile = open(outputdir + '/predicted_ids_' + str(x) + '.txt', 'w')
        for ID in predicted_ids:
            IDsfile.write(ID + '\n')
        IDsfile.close()

    return predicted_pos


def adjust_seq_positions(extracted_seq, outputDir, idProc, max_len_threshold, min_len_threshold, tg_ca, TSD):
    seq1file = open(outputDir + '/splittedChrWindow_' + str(idProc) + '.fasta', 'w')
    iterSeq = one_hot2fasta(extracted_seq)
    seq1file.write('>seq_1\n' + iterSeq + '\n')
    seq1file.close()

    try:
        # execute LTR_Finder in orde to find start and end positions of the LTR-RTs
        if tg_ca:
            finder_filter = '1111'
        else:
            finder_filter = '0000'
        if TSD:
            finder_filter += '1'
        else:
            finder_filter += '0'
        finder_filter += '000000'

        output = subprocess.run(
            ['ltr_finder', '-F', finder_filter, '-D', str(max_len_threshold), '-d', str(min_len_threshold), '-w2', '-C', '-p', '20', '-M', '0.80',
             outputDir + '/splittedChrWindow_' + str(idProc) + '.fasta'], stdout=subprocess.PIPE, text=True)

    except Exception as e:
        print("FATAL ERROR. LTR_finder could not be executed, please re-execute Inpactor2...")
        print(e)
        return []

    bestHits = []
    if "No LTR Retrotransposons Found" not in output.stdout:
        hits = output.stdout.split('\n')

        # to avoid the six first and the last two lines of LTR_finder results
        for hit in hits[6:-3]:
            columns = hit.split('\t')
            element_int = int(columns[2].split('-')[0])
            element_end = int(columns[2].split('-')[1])
            bestHits.append([element_int, element_end])

    try:
        os.remove(outputDir + '/splittedChrWindow_' + str(idProc) + '.fasta')
    except:
        print("I could not delete the file: " + outputDir + "/splittedChrWindow_" + str(idProc) + ".fasta")
    return bestHits


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
This function calculates k-mer frequencies using a CNN
"""
def Inpactor2_kmer(position_detected, batch_size, splitted_genome_ltr):
    extracted_sequences = np.zeros(
        (len(position_detected), splitted_genome_ltr.shape[1], splitted_genome_ltr.shape[2]), dtype=np.uint8)
    for i in range(len(position_detected)):
        init = int(position_detected[i][0])
        end = int(position_detected[i][1])
        window_index = int(position_detected[i][2])
        extracted_sequences[i, :, init:end] = splitted_genome_ltr[window_index, :, init:end]

    kmer_extractor = kmer_extractor_model(extracted_sequences)
    kmer_counts = kmer_extractor.predict(extracted_sequences, batch_size=batch_size)
    return kmer_counts


"""
This function uses the FNN to automatically filter non-intact sequences.
"""
def Inpactor2_Filter(kmer_counts, ids_predicted, positions_detected, filter_threshold):
    new_position_predicted = []
    new_ids_predicted = []
    installation_path = os.path.dirname(os.path.realpath(__file__))
    # Scaling
    scaling_path = installation_path + '/Models/std_scaler_filter.bin'
    scaler = load(scaling_path)
    feature_vectors_scaler = scaler.transform(kmer_counts)

    # PCA
    pca_path = installation_path + '/Models/std_pca_filter.bin'
    pca = load(pca_path)
    features_pca = pca.transform(feature_vectors_scaler)

    # loading DNN model and predict labels (lineages)
    model_path = installation_path + '/Models/Inpactor_Filter.hdf5'
    model = tf.keras.models.load_model(model_path, custom_objects={'f1_m': f1_m})
    predictions = model.predict(features_pca)
    binary_predictions = [argmax(x) for x in predictions]
    filtered_elements = np.zeros((len([x for x in range(len(binary_predictions)) if binary_predictions[x] == 0 and predictions[x, 0] > filter_threshold]), kmer_counts.shape[1]), dtype=np.int16)

    j = 0  # index of filtered_elements array
    for i in range(len(binary_predictions)):
        if binary_predictions[i] == 0 and predictions[i, 0] > filter_threshold:
            filtered_elements[j, :] = kmer_counts[i, :]
            new_position_predicted.append(positions_detected[i])
            new_ids_predicted.append(ids_predicted[i] + "#" + str(predictions[i, 0]))
            j += 1
    return filtered_elements, new_ids_predicted, new_position_predicted


"""
This function predicts the lineage of each sequence in the k-mer file using a pre-trained DNN
"""
def Inpactor2_Class(seq_data):
    installation_path = os.path.dirname(os.path.realpath(__file__))
    lineages_names_dic = {0: 'Negative', 1: 'ALE/RETROFIT', 3: 'ANGELA', 4: 'BIANCA', 8: 'IKEROS', 9: 'IVANA/ORYCO',
                          11: 'TAR', 12: 'TORK', 13: 'SIRE', 14: 'CRM', 16: 'GALADRIEL', 17: 'REINA', 18: 'TEKAY/DEL',
                          19: 'ATHILA', 20: 'TAT'}

    # Scaling
    scaling_path = installation_path + '/Models/std_scaler.bin'
    scaler = load(scaling_path)
    feature_vectors_scaler = scaler.transform(seq_data)

    # PCA
    pca_path = installation_path + '/Models/std_pca.bin'
    pca = load(pca_path)
    features_pca = pca.transform(feature_vectors_scaler)

    # loading DNN model and predict labels (lineages)
    model_path = installation_path + '/Models/Inpactor_Class.hdf5'
    model = tf.keras.models.load_model(model_path, custom_objects={'f1_m': f1_m})
    predictions = model.predict(features_pca)
    lineages_ids = [argmax(x) for x in predictions]
    perc_list = []
    for i in range(predictions.shape[0]):
        perc_list.append(predictions[i, lineages_ids[i]])
    return [lineages_names_dic[x] for x in lineages_ids], perc_list

def non_maximal_suppression(ids_predicted, predictions, percentages, iou_threshold, curation, predicted_ltr_rts):
    finalIds = []
    for i in range(len(ids_predicted)):
        # to create the final IDs
        finalIds.append(ids_predicted[i] + "#" + predictions[i])

    deleted_seqs = []
    for i in range(len(finalIds)):
        if i not in deleted_seqs:
            cluster = [i]
            for j in range(len(finalIds)):
                if i != j and j not in deleted_seqs and iou(finalIds[i], finalIds[j]) > iou_threshold:
                    cluster.append(j)

            if len(cluster) > 1:  # only clusters with more than one seq
                # to search the best score in the cluster
                best_score = 0
                pos_best = -1
                for member in cluster:
                    columns = finalIds[member].split("#")
                    perClass = percentages[member]
                    percDect = float(columns[3])
                    if curation:
                        percFilt = float(columns[4])
                        member_score = (percDect + percFilt + perClass) / 3
                    else:
                        member_score = (percDect + perClass) / 2

                    if member_score > best_score:
                        best_score = member_score
                        pos_best = member

                #to add all non-max predictions to array
                deleted_seqs.extend([x for x in cluster if x != pos_best])

    # to delete all non-max predictions
    new_percentages = [percentages[x] for x in range(len(percentages)) if x not in deleted_seqs]
    new_predictions = [predictions[x] for x in range(len(predictions)) if x not in deleted_seqs]
    finalIds = [finalIds[x] for x in range(len(finalIds)) if x not in deleted_seqs]
    new_ltr_predicted = [predicted_ltr_rts[x] for x in range(len(predicted_ltr_rts)) if x not in deleted_seqs]

    return finalIds, new_predictions, new_percentages, new_ltr_predicted


def iou(seqX, seqY):
    columns = seqX.split("#")
    idseqX = columns[0]
    initPosX = int(columns[1])
    endPosX = int(columns[2])

    columns = seqY.split("#")
    idseqY = columns[0]
    initPosY = int(columns[1])
    endPosY = int(columns[2])

    if idseqX == idseqY:
        intersection = max(0, min(endPosX, endPosY) - max(initPosX, initPosY))
        union = max(0.001, max(endPosX, endPosY) - min(initPosX, initPosY))
        return intersection / union
    else:
        return 0


"""
This function predicts the lineage of each sequence in the k-mer file using a pre-trained DNN
"""
def create_bed_file(finalIds, percentajes, outputDir, curation):
    # to write the results into a bed file
    f = open(outputDir + '/Inpactor2_predictions.tab', 'w')
    i = 0
    for seqid in finalIds:
        columns = seqid.split("#")
        idseq = columns[0]
        initPos = columns[1]
        endPos = columns[2]
        percDect = columns[3]
        perClass = str(percentajes[i])
        if curation:
            percFilt = columns[4]
            lineage = columns[5]
            f.write(idseq + '\t' + initPos + '\t' + endPos + '\t' + str(
                int(endPos) - int(
                    initPos)) + '\t' + lineage + '\t' + percDect + '\t' + percFilt + '\t' + perClass + '\n')
        else:
            lineage = columns[4]
            f.write(idseq + '\t' + initPos + '\t' + endPos + '\t' + str(
                int(endPos) - int(initPos)) + '\t' + lineage + '\t' + percDect + '\t-\t' + perClass + '\n')
        i += 1
    f.close()
    return finalIds


def create_fasta_file_master(finalIds, threads, ltr_predicted_final, outputDir, curation):
    n = len(finalIds)
    seqs_per_procs = int(n / threads)
    remain = n % threads

    splitted_sequences_list = []
    for i in range(threads):
        if i < remain:
            init = i * (seqs_per_procs + 1)
            end = init + seqs_per_procs + 1
        else:
            init = i * seqs_per_procs + remain
            end = n if init + seqs_per_procs > n else init + seqs_per_procs
        splitted_sequences_list.append(ltr_predicted_final[init:end])
    ltr_predicted_final = None  # clean unusable variable

    pool = multiprocessing.Pool(processes=threads)
    localresults = [pool.apply_async(create_fasta_file_slave,
                                     args=[splitted_sequences_list[x], finalIds, x, seqs_per_procs, n, remain,
                                           outputDir, curation]) for x in range(threads)]
    localSequences = [p.get() for p in localresults]
    outputFile = open(outputDir + '/Inpactor2_library.fasta', 'w')
    for i in range(threads):
        filei = open(outputDir + '/Inpactor2_library_' + str(i) + '.fasta', 'r')
        lines = filei.readlines()
        for line in lines:
            outputFile.write(line)
        filei.close()
        try:
            os.remove(outputDir + '/Inpactor2_library_' + str(i) + '.fasta')
        except:
            print('I cannot delete the file: ' + outputDir + '/Inpactor2_library_' + str(i) + '.fasta')
    outputFile.close()
    pool.close()

"""
This function takes the joined prediction and creates a fasta file containing
all LTR retrotransposon's sequences
"""
def create_fasta_file_slave(predicted_ltr_rts, finalIds, x, seqs_per_procs, n, remain, outputDir, curation):
    res = ""
    i = 0
    result_file = open(outputDir + '/Inpactor2_library_' + str(x) + '.fasta', 'w')
    if x < remain:
        init = x * (seqs_per_procs + 1)
        end = init + seqs_per_procs + 1
    else:
        init = x * seqs_per_procs + remain
        end = n if init + seqs_per_procs > n else init + seqs_per_procs

    while init < end and init < len(finalIds):
        p = finalIds[init]
        columns = p.split("#")
        if curation:
            lineage = columns[5]
        else:
            lineage = columns[4]

        idseq = columns[0]
        initPos = columns[1]
        endPos = columns[2]
        results = '>' + idseq + '_' + initPos + '_' + endPos + '#LTR/' + lineage.replace('/', '-') + '\n' + \
                  predicted_ltr_rts[i] + '\n'
        result_file.write(results)
        init += 1
        i += 1
    return res


if __name__ == '__main__':
    print("\n#########################################################################")
    print("#                                                                       #")
    print("# Inpactor2: A software based on deep learning to identify and classify #")
    print("#               LTR-retrotransposons in plant genomes                   #")
    print("#                                                                       #")
    print("#########################################################################\n")

    ### read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, dest='fasta_file', help='Fasta file containing DNA sequences')
    parser.add_argument('-o', '--output-dir', required=False, dest='outputDir', help='Path of the output directory')
    parser.add_argument('-t', '--threads', required=False, dest='threads',
                        help='Number of threads to be used by Inpactor2')
    parser.add_argument('-a', '--annotate', required=False, dest='annotate',
                        help='Annotate LTR retrotransposons? [yes or not]')
    parser.add_argument('-m', '--max-len', required=False, dest='max_len_threshold',
                        help='Maximum length for detecting LTR-retrotransposons [1-50000]')
    parser.add_argument('-n', '--min-len', required=False, dest='min_len_threshold',
                        help='Minimum length for detecting LTR-retrotransposons [1-50000]')
    parser.add_argument('-i', '--tg-ca', required=False, dest='tg_ca',
                        help='Keep only elements with TG-CA-LTRs? [yes or not]')
    parser.add_argument('-d', '--tsd', required=False, dest='TSD',
                        help='Keep only elements with TDS? [yes or not]')
    parser.add_argument('-c', '--curated', required=False, dest='curation',
                        help='keep on only intact elements? [yes or not]')
    parser.add_argument('-C', '--cycles', required=False, dest='cycles',
                        help='Number of analysis cycles [1-5]')
    parser.add_argument('-V', '--verbose', required=False, dest='verbose',
                        help='activate verbose? [yes or not]')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v1.0')

    options = parser.parse_args()
    file = options.fasta_file
    outputDir = options.outputDir
    threads = options.threads
    annotate = options.annotate
    max_len_threshold = options.max_len_threshold
    min_len_threshold = options.min_len_threshold
    tg_ca = options.tg_ca
    TSD = options.TSD
    curation = options.curation
    cycles = options.cycles
    verbose = options.verbose

    ##############################################################################
    # Parameters' validation
    if file is None:
        print('FATAL ERROR: Missing fasta file parameter (-f or --file). Exiting')
        sys.exit(0)
    elif not os.path.exists(file):
        print('FATAL ERROR: fasta file did not found at path: ' + file)
        sys.exit(0)
    if outputDir is None:
        outputDir = os.path.dirname(os.path.realpath(__file__))
        print("WARNING: Missing output directory, using by default: " + outputDir)
    elif not os.path.exists(outputDir):
        print('FATAL ERROR: output directory did not found at path: ' + outputDir)
        sys.exit(0)
    if threads is None or threads == -1:
        threads = int(psutil.cpu_count())
        print("WARNING: Missing threads parameter, using by default: " + str(threads))
    else:
        threads = int(threads)
    if annotate is None:
        annotate = 'yes'
        print("WARNING: Missing annotation parameter (-a or --annotate), using by default: " + str(annotate))
    elif annotate.upper() not in ['YES', 'NO']:
        print('FATAL ERROR: unknown value of -a parameter: ' + annotate + '. This parameter must be yes or no')
        sys.exit(0)
    if max_len_threshold is None:
        max_len_threshold = 28000
        print("WARNING: Missing max length parameter, using by default: 28000")
    elif int(max_len_threshold) > 50000 or int(max_len_threshold) < 1:
        print('FATAL ERROR: max length parameter must be between 1 and 50000')
        sys.exit(0)
    else:
        max_len_threshold = int(max_len_threshold)
    if min_len_threshold is None:
        min_len_threshold = 2000
        print("WARNING: Missing min length parameter, using by default: 2000")
    elif int(min_len_threshold) > 50000 or int(min_len_threshold) < 1:
        print('FATAL ERROR: min length parameter must be between 1 and 50000')
        sys.exit(0)
    else:
        min_len_threshold = int(min_len_threshold)
    if tg_ca is None:
        tg_ca = True
        print("WARNING: Missing TG-CA filter parameter, using by default: yes")
    elif tg_ca.upper() not in ['YES', 'NO']:
        print('FATAL ERROR: unknown value of -i parameter: ' + tg_ca + '. This parameter must be yes or no')
        sys.exit(0)
    else:
        if tg_ca.upper() == 'YES':
            tg_ca = True
        else:
            tg_ca = False
    if TSD is None:
        TSD = True
        print("WARNING: Missing TSD mismatch number parameter, using by default: yes")
    elif TSD.upper() not in ['YES', 'NO']:
        print('FATAL ERROR: unknown value of -d parameter: ' + TSD + '. This parameter must be yes or no')
        sys.exit(0)
    else:
        if TSD.upper() == 'YES':
            TSD = True
        else:
            TSD = False
    if curation is None:
        curation = True
        print("WARNING: Missing curation parameter, using by default: yes")
    elif curation.upper() not in ['YES', 'NO']:
        print('FATAL ERROR: unknown value of -c parameter: ' + curation + '. This parameter must be yes or no')
        sys.exit(0)
    else:
        if curation.upper() == 'YES':
            curation = True
        else:
            curation = False
    if cycles is None:
        cycles = 1
        print("WARNING: Missing cycles parameter, using by default: 1")
    elif int(cycles) > 5 or int(cycles) < 1:
        print('FATAL ERROR: cycle number must be between 1 and 5')
        sys.exit(0)
    else:
        cycles = int(cycles)
    if verbose is None:
        verbose = False
    elif verbose.upper() not in ['YES', 'NO']:
        print('FATAL ERROR: unknown value of -V parameter: ' + curation + '. This parameter must be yes or no')
        sys.exit(0)
    else:
        if verbose.upper() == 'YES':
            verbose = True
        else:
            verbose = False
    ##################################################################################
    # global configuration variables
    total_win_len = 50000
    batch_size = 2
    iou_threshold = 0.6
    detec_threshold = 0.6
    filter_threshold = 0.6

    ##################################################################################
    # Start of detection cycles
    slide_win = int(total_win_len / cycles)
    total_time = []
    finalIds_cycles = []
    predictions_cycles = []
    percentages_cycles = []
    ltr_predicted_final_cycles = []
    for cycle in range(0, cycles):
        print('---------------------------------------------------------------------------')
        print('INFO: Doing cycle # ' + str(cycle + 1))
        slide = slide_win * cycle
        ##################################################################################
        # First step: Split input sequences into chunks of 50k bp and convert it into one-hot coding
        tf.keras.backend.clear_session()  # to clean GPU memory
        print('INFO: Splitting input sequences into chunks of size ' + str(
            total_win_len) + ' and converting them into one-hot coding ...')
        start = time.time()
        list_ids, list_seqs = get_final_dataset_size(file, total_win_len, slide)

        # To validate that sequences only contain valid DNA nucleotides in parallel
        check_nucleotides_master(list_seqs, threads)

        # Run in parallel the splitter
        splitted_genome = create_dataset_master(list_ids, list_seqs, threads, total_win_len, outputDir)
        list_seqs = None  # to clean unusable variable

        finish = time.time()
        total_time.append(finish - start)
        print('INFO: Splitting of input sequences done!!!! [time=' + str(finish - start) + ']')

        ##################################################################################
        # Second step: Predict initial and end position of LTR-RTs in each chunk
        print('INFO: Predicting which genome chunk contains LTR-RTs...')
        start = time.time()
        splitted_genome_ltr, detection_proba, list_ids = Inpactor2_Detect(splitted_genome, detec_threshold, list_ids)

        if verbose:
            print("------ Verbose")
            print("\tWindows detected with LTR-retrotransposons inside: " + str(
                splitted_genome_ltr.shape[0]) + " of " + str(splitted_genome.shape[0]))
            print("------ ")

        splitted_genome = None  # to clean unusable variable
        finish = time.time()
        total_time.append(finish - start)
        print('INFO: LTR-RTs containing prediction done!!!! [time=' + str(finish - start) + ']')

        ##################################################################################
        # Third step: Extract sequences predicted as LTR-RTs
        print('INFO: Extracting sequences predicted as LTR-RTs ...')
        # Run in parallel the extraction
        pos_predicted, ids_predicted = sequences_extractor_master(splitted_genome_ltr, threads, outputDir,
                                                                  max_len_threshold, min_len_threshold, list_ids, tg_ca,
                                                                  TSD, detection_proba)

        splitted_genome_list = None  # to clean unusable variable
        detection_proba = None  # to clean unusable variable

        if verbose:
            print("------ Verbose")
            print("\tNumber of LTR-retrotransposons detected: " + str(len(pos_predicted)))
            print("------ ")

        finish = time.time()
        total_time.append(finish - start)
        print('INFO: Extraction done!!!! [time=' + str(finish - start) + ']')

        if len(pos_predicted) == 0:
            print('WARNING: There is no LTR retrotransposons that satisfy the conditions after structural filtration, '
                  'try modifying the parameters -m, -n, -i, and -d ....')
            sys.exit(0)

        ##################################################################################
        # Fourth step: k-mer Counting (1<=k<=6) from sequences using a DNN
        print('INFO: Counting k-mer frequencies using a DNN ...')
        start = time.time()
        kmer_counts = Inpactor2_kmer(pos_predicted, batch_size, splitted_genome_ltr)
        finish = time.time()
        total_time.append(finish - start)
        print('INFO: K-mer counting done!!!! [time=' + str(finish - start) + ']')

        ##################################################################################
        # Fifth step: Filter sequences that are not full-length with a FNN.
        if curation:
            print('INFO: Filtering non-intact LTR-retrotransposons ...')
            start = time.time()
            filtered_seqs, ids_predicted, new_pos_predicted = Inpactor2_Filter(kmer_counts, ids_predicted,
                                                                               pos_predicted, filter_threshold)
            if verbose:
                print("------ Verbose")
                print("\tNumber of LTR-retrotransposons filtered: " + str(
                    len(pos_predicted) - len(new_pos_predicted)) + " of " + str(len(pos_predicted)))
                print("------ ")

            kmer_counts = None  # clean unusable variable
            pos_predicted = None  # clean unusable variable

            finish = time.time()
            total_time.append(finish - start)
            print('INFO: Filtering done!!!! [time=' + str(finish - start) + ']')

            if filtered_seqs.shape[0] == 0:
                print(
                    'WARNING: There is no LTR retrotransposons that satisfy the conditions after curation, try to re-run '
                    'Inpactor2 with the option -c no ....')
                sys.exit(0)
        else:
            filtered_seqs = kmer_counts
            new_pos_predicted = pos_predicted
            kmer_counts = None  # clean unusable variable

        ##################################################################################
        # Sixth step: Predict the lineage from the pre-trained DNN.
        print('INFO: Predicting the lineages from sequences ...')
        start = time.time()
        predictions, percentages = Inpactor2_Class(filtered_seqs)
        finish = time.time()
        total_time.append(finish - start)
        print('INFO: Prediction of the lineages from sequences done!!! [time=' + str(finish - start) + ']')

        ##################################################################################
        # Join cycle results and save them
        print('INFO: Saving cycle results ...')
        start = time.time()
        finalIds_cycles.extend(ids_predicted)
        predictions_cycles.extend(predictions)
        percentages_cycles.extend(percentages)

        for i in range(len(new_pos_predicted)):
            init = int(new_pos_predicted[i][0])
            end = int(new_pos_predicted[i][1])
            window_index = int(new_pos_predicted[i][2])
            ltr_predicted_final_cycles.append(one_hot2fasta(splitted_genome_ltr[window_index, :, init:end]))

        filtered_seqs = None  # clean unusable variable
        new_ltr_predicted = None  # clean unusable variable
        finalIds = None  # clean unusable variable
        predictions = None  # clean unusable variable
        ids_predicted = None  # clean unusable variable
        percentages = None  # clean unusable variable
        splitted_genome_ltr = None  # clean unusable variable

        finish = time.time()
        total_time.append(finish - start)
        print('INFO: Cycle results saved!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # End of cycles

    ##################################################################################
    # Seventh step: Applying Non-maximal suppression
    print('INFO: Suppressing non-maximal predictions...')
    start = time.time()
    finalIds, predictions, percentages, ltr_predicted_final = non_maximal_suppression(finalIds_cycles, predictions_cycles,
                                                                                      percentages_cycles, iou_threshold,
                                                                                      curation, ltr_predicted_final_cycles)
    if verbose:
        print("------ Verbose")
        print("\tNumber of LTR-retrotransposons removed: " + str(
            len(ltr_predicted_final_cycles) - len(ltr_predicted_final)) + " of " + str(len(ltr_predicted_final_cycles)))
        print("------ ")
    finish = time.time()
    total_time.append(finish - start)
    print('INFO: Non-max suppression done!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Eighth step: Creating the description file of the predictions made by the DNNs
    print('INFO: Creating the prediction descriptions file...')
    start = time.time()
    create_bed_file(finalIds, percentages, outputDir, curation)
    predictions = None  # clean unusable variable
    ids_predicted = None  # clean unusable variable
    percentages = None  # clean unusable variable
    finish = time.time()
    total_time.append(finish - start)
    print('INFO: Creating output file done!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # nineth step: Creating fasta file with LTR retrotransposons classified
    print('INFO: Creating LTR-retrotransposon library...')
    start = time.time()

    create_fasta_file_master(finalIds, threads, ltr_predicted_final, outputDir, curation)

    finalIds = None  # clean unusable variable
    splitted_sequences_list = None  # clean unusable variable
    localSequences = None  # clean unusable variable
    finalIds_cycles = None  # clean unusable variable
    predictions_cycles = None  # clean unusable variable
    percentages_cycles = None  # clean unusable variable
    ltr_predicted_final_cycles = None  # clean unusable variable

    finish = time.time()
    total_time.append(finish - start)
    print('INFO: Library created!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # tenth step: Annotating LTR-RTs using RepeatMasker
    if annotate.upper() == 'YES':
        print('INFO: Annotating LTR-retrotranposons with RepeatMasker...')
        start = time.time()
        result_command = os.system(
            'RepeatMasker -pa ' + str(threads) + ' -lib ' + outputDir + '/Inpactor2_library.fasta '
                                                                        '-dir ' + outputDir + ' -gff -nolow -no_is -norna ' + file)
        if result_command != 0:
            print('FATAL ERROR: RepeatMasker failed!!!')
        else:
            finish = time.time()
            total_time.append(finish - start)
            print('INFO: Annotation done!!! [time=' + str(finish - start) + ']')

    print('INFO: Inpactor2 execution done successfully [total time=' + str(sum(total_time)) + ']')
