#!/usr/bin/python3


import librosa
import subprocess
import csv
import os
import argparse
from FeatureExtraction import FeatureExtraction
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed


#FEATURE = "timeseries"
FEATURE = "mfcc"


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filelist", help="Name of the task")
    parser.add_argument("audio_dir", help="The top directory of the file to be copied")
    parser.add_argument("feature_dir", help="The top directory of the file at destination")
    parser.add_argument("--ext", help="Extention the files if there are any", default=".mp3")
    args = parser.parse_args()
    return args


def read_and_write_features(input_output: tuple):
    if os.path.isfile(input_output[1]):
        return
    fe = FeatureExtraction(input_output[0])
    #fe.time_series()
    #fe.write_feature(input_output[1])
    fe.mfcc()
    fe.write_feature(input_output[1])
    
    return


def main():
    args = read_args()
    with open(args.filelist, newline='') as f:
        csvreader = csv.reader(f, delimiter='\t')
        allfiles = [row[0] for idx, row in enumerate(csvreader)]
        alldir = set([os.path.dirname(x) for x in allfiles])
        print("Creating subdirectories...")
        for dir in alldir:
            absdir = os.path.join(args.feature_dir, dir)
            retcode = subprocess.call(["mkdir", "-p", absdir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        audio_abs_paths = [os.path.join(args.audio_dir, f+args.ext) for f in allfiles]
        feature_abs_paths = [os.path.join(args.feature_dir, f+"."+FEATURE) for f in allfiles]
        print("Generating features...")
        with ProcessPoolExecutor(max_workers = 3) as executor:
            executor.map(read_and_write_features, zip(audio_abs_paths, feature_abs_paths))

    return


if __name__ == "__main__":
    main()






