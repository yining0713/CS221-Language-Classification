#!/usr/bin/python3


import random
import configparser
import argparse
import os
import csv


def generate_random_numbers(total_number: int):
    """
    Randomize the order
    """
    random_order = random.sample(range(total_number), total_number)
    return random_order


def pick_valid_audio(row, language_code, country):
    if row[8] == language_code and \
        (row[7] == country or row[7] == ''):
        return True
    return False


def remove_extension(file_path):
    exts = [".mp3", ".wav"]
    for ext in exts:
        if file_path.endswith(ext):
            return file_path[:-len(ext)]
    return file_path


def generate_training_data(language, output_file_training, output_file_testing, config, data_dir):
    """
    Extract certain number of audios from a language,
    Write to an output file
    """
    language_code = config["LANGUAGE_CODE"][language]
    country = config["COUNTRY"][language]
    num_audio_train = int(config["TRAINING_SIZE"][language])
    num_audio_test = int(config["TESTING_SIZE"][language])
    total_num_audio = num_audio_train + num_audio_test
    valid_file = os.path.join(data_dir, language, "validated.tsv")
    
    selected_audios_train = []
    selected_audios_test = []
    num_valid_audios = 0
    with open(valid_file, newline='') as f:
        csvreader = csv.reader(f, delimiter='\t')
        allrows = [row for idx, row in enumerate(csvreader)]
        num_audio_all = len(allrows)-1
        selected_row_number = [x+1 for x in generate_random_numbers(num_audio_all)]
        
        for num in selected_row_number:
            if pick_valid_audio(allrows[num], language_code, country):
                num_valid_audios += 1
                if num_valid_audios <= num_audio_train:
                    selected_audios_train.append(remove_extension(allrows[num][1]))
                elif num_valid_audios <= total_num_audio:
                    selected_audios_test.append(remove_extension(allrows[num][1]))
                else:
                    break
        if num_valid_audios < total_num_audio:
            print(f"WARNING: Not enough valid training files for {language}.")

    write_audio(output_file_training, selected_audios_train, language)
    write_audio(output_file_testing, selected_audios_test, language)
    
    return


def write_audio(output_file, audio_list, language):
    with open(output_file, 'a+', newline='') as fs:
        csvwriter = csv.writer(fs, delimiter='\t', lineterminator='\n')
        for audio in audio_list:
            filepath = os.path.join(language, 'clips', audio)
            csvwriter.writerow([filepath, language])
    return
    

def read_config(config_file):
    config  = configparser.ConfigParser()
    config.read(config_file)
    return config


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Name of the task")
    args = parser.parse_args()
    return args

def main():
    """
    Select certain number of audios from each language
    Populate a file for training
    With Labeling:
    <audio_file>\t<language_code>
    """
    config_file = "../config/training.ini"
    config = read_config(config_file)
    args = read_args()

    data_dir = config["FILE_DIR"]["Data_dir"]
    output_dir = config["FILE_DIR"]["Data_prepare_dir"]
    output_file_training = os.path.join(output_dir, args.task+'.training.txt')
    output_file_testing = os.path.join(output_dir, args.task+'.testing.txt')

    if os.path.isfile(output_file_training):
        print(f"Amending existing file: {output_file_training}")
    if os.path.isfile(output_file_testing):
        print(f"Amending existing file: {output_file_testing}")

    print(f"output to {output_file_training} and {output_file_testing}")
    for language in config["TRAINING_SIZE"]:
        print(f"Processing files from {language}.")
        generate_training_data(language, output_file_training, output_file_testing, config, data_dir)

    return


if __name__ == '__main__':
    main()
