import json
import os
import math
import csv
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

TRACKS_FILE = '/dev/github/year-guesser/data/tracks.psv'
OUTPUT_FILE = '/dev/github/year-guesser/data/output.json'
SAMPLE_RATE = 22050
HOP_LENGTH = 512
NUM_FFT = 2048
NUM_MFCC = 13
TARGET_DURATION = 120

def get_tracks():
    tracks =  []
 
    with open(TRACKS_FILE, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='|')

        for row in reader:
            tracks.append({ "filename": row[0], "year": row[1] })

    return tracks

def analyze_tracks(tracks):
    
    outputs = {
        "filenames": [],
        "labels": [],
        "mfccs": []
    }

    for track in tracks:
        year = track["year"]
        filename = track["filename"]
        
        if year == 0:
            continue;

        print(filename);
        signal, sr = librosa.load(filename, sr=SAMPLE_RATE)

        duration = librosa.get_duration(y=signal, sr=sr)

        if duration < TARGET_DURATION:
            continue;

        samples_per_track = SAMPLE_RATE * TARGET_DURATION
       
        first_sample = int((len(signal) - samples_per_track) / 2) 
        start = first_sample
        finish = first_sample + samples_per_track
        mfccs = librosa.feature.mfcc(y=signal[start:finish], sr=sr, n_mfcc=NUM_MFCC, n_fft=NUM_FFT, hop_length=HOP_LENGTH)
        mfccs = mfccs.T
        #delta_mfccs = librosa.feature.delta(mfccs)
        #delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        # mfccs_features = np.concatenate(mfccs, delta_mfccs, delta2_mfccs)
        # don't know why this ^ fails

        outputs["labels"].append(year)
        outputs["filenames"].append(filename)
        outputs["mfccs"].append(mfccs.tolist())
        
        # write out to csv not json
        # do you just have a ton of samples all labeld with the year?
        #if len(mfccs_features) == num_mfcc_vectors_per_segment:                

        #    outputs["labels"].append(track["year"])
        #    outputs["mfccs"].append(mfccs_features.tolist())
            
    with open(OUTPUT_FILE, "w") as fp:
        json.dump(outputs, fp)

        
if __name__ == "__main__":
    tracks = get_tracks()
    output = analyze_tracks(tracks)

    #print(output)
    #save_mfcc(DATASET_FILE, OUTPUT_FILE)
