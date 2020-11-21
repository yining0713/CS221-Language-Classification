import numpy as np
import configparser
import csv
import os
import sys
from Log import logging, Log

target_values = {
    "german": 1,
    "english": 2,
    "arabic": 3,
    "mandarin": 4,
    "ukrainian": 5
}

def load_time_series(file):
    """Load features into a one dimentional numpy array
    """
    features = []
    with open(file, 'r') as f:
        line = f.readline().strip()
        while line:
            features.append(float(line))
            line = f.readline().strip()
    features = np.asarray(features)
    return features


def create_vector(np_array, end, file, start=0):
    if np_array.size < end:
        logging.writelog(f"Error: {file} too short, skipping..." + "\n")
        return None
    return np_array[start:end]


def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    logging.writelog("Training data:\n")
    for language in config["TRAINING_SIZE"]:
        size = config["TRAINING_SIZE"][language]
        logging.writelog(f"{language}: {size}" + '\n')

    return config

def numpy_array_to_string(np_array):
    return np.array2string(np_array, precision=8, separator=',').replace('\n', '')[1:-1]

def populate_vector_table(language_list, 
                            file_list, 
                            feature_dir, 
                            value_file,
                            target_file):
    with open(file_list, newline='') as csvfile, \
        open(value_file, 'w+') as v, \
        open(target_file, 'w+') as t:
        print(f'Reading {file_list}')
        csvreader = csv.reader(csvfile, delimiter='\t')
        for row in csvreader:
            if row[1] in language_list:
                feature_file_path = os.path.join(feature_dir, row[0]+'.mfcc')
                if not os.path.isfile(feature_file_path):
                    logging.writelog(f"{feature_file_path} does not exist, skipping..." + '\n')
                    continue
                all_features = load_time_series(feature_file_path)
                selected_features = create_vector(all_features, 20, feature_file_path)
                if selected_features is None:
                    print("Too short")
                    continue
                feature_in_string = numpy_array_to_string(selected_features)
                v.write(feature_in_string + '\n')
                target_value = target_values[row[1]]
                t.write(str(target_value) + '\n')
    return


def write_training():

    topdir = os.path.split(os.path.split(os.path.realpath(sys.argv[0]))[0])[0]
    config = read_config(os.path.join(topdir, "config/training.ini"))
    #train_file_list = os.path.join(topdir, "training/prepare_data/20201108_6_10000.training.txt.temp")
    train_file_list = os.path.join(topdir, config["FILE_DIR"]["training_file"])
    test_file_list = os.path.join(topdir, config["FILE_DIR"]["testing_file"])
    languages_in_training = list(config["TRAINING_SIZE"].keys())

    populate_vector_table(languages_in_training, 
                            train_file_list, 
                            os.path.join(topdir, config["FILE_DIR"]["feature_dir"]), 
                            os.path.join(topdir, 'log/mfcc_training_value.csv'),
                            os.path.join(topdir, 'log/mfcc_training_target.csv'))
    populate_vector_table(languages_in_training, 
                            test_file_list, 
                            os.path.join(topdir, config["FILE_DIR"]["feature_dir"]), 
                            os.path.join(topdir, 'log/mfcc_testing_value.csv'),
                            os.path.join(topdir, 'log/mfcc_testing_target.csv'))

    return



if __name__ == '__main__':
    write_training()


# def load_all_data_train():
#     feature_arabic_1 = load_time_series("../data/mock_feature/arabic/001.txt")
#     feature_english_1 = load_time_series("../data/mock_feature/english/003.txt")
#     feature_german_1 = load_time_series("../data/mock_feature/german/005.txt")
#     feature_mandarin_1 = load_time_series("../data/mock_feature/mandarin/007.txt")
#     feature_ukrainian_1 = load_time_series("../data/mock_feature/ukrainian/009.txt")
#     training_array = np.array([feature_arabic_1, 
#                             feature_english_1,
#                             feature_german_1,
#                             feature_mandarin_1,
#                             feature_ukrainian_1])
#     training_array_label = np.array([1,2,3,4,5])
#     feature_arabic_test = load_time_series("../data/mock_feature/arabic/002.txt")
#     feature_english_test = load_time_series("../data/mock_feature/english/004.txt")
#     feature_german_test = load_time_series("../data/mock_feature/german/006.txt")
#     feature_mandarin_test = load_time_series("../data/mock_feature/mandarin/008.txt")
#     feature_ukrainian_test = load_time_series("../data/mock_feature/ukrainian/010.txt")
#     testing_array = np.array([feature_arabic_test, 
#                             feature_english_test,
#                             feature_german_test,
#                             feature_mandarin_test,
#                             feature_ukrainian_test])
#     testing_array_label = np.array([1,2,3,4,5])
#     return training_array, training_array_label, testing_array, testing_array_label

###############
#def load_all_data(training_config, top_dir):
#    """
#    Select certain number of audios from each language
#    Populate a file for training
#    With Labeling:
#    <audio_file>\t<language_code>
#    """
#    config = read_config(training_config)
#
#    languages = config["TRAINING_SIZE"]
#    for l in languages:
#        
#
#    return
