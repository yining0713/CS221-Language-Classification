#!/usr/bin/python3


import librosa
import sys
import numpy as np
import sklearn

class FeatureExtraction:
    def __init__(self, audio_path):
        self.raw = librosa.load(audio_path)
        self.features = ''

    def time_series(self):
        self.features = self.raw[0]

    def write_feature(self, output_file):
        with open(output_file, 'w+') as f:
            for i in self.features:
                f.write(str(i) + '\n')
    
    def mfcc(self):
        x, r = self.raw
        mfcc_data = librosa.feature.mfcc(x, sr=r)
        # self.features = mfcc_data
        self.features = sklearn.preprocessing.scale(mfcc_data, axis=1).mean(axis=1)


        



def main():
    input_audio = sys.argv[1]
    output_file = sys.argv[2]
    fe = FeatureExtraction(input_audio)
    fe.mfcc()
    #print(fe.features.shape)
    #print(fe.features)
    fe.write_feature(output_file)
    return


if __name__ == '__main__':
    main()
