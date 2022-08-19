import json
import os
import math
import csv
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import keras.api._v2.keras as keras

#INPUT_FILE = '/Buggles.mp3'
INPUT_FILE = '/The-Kinks-Lola-.mp3'
OUTPUT_FILE = './data/example.json'
MODEL_FILE = './models/1.model'
SAMPLE_RATE = 22050
HOP_LENGTH = 512
NUM_FFT = 2048
NUM_MFCC = 13
TARGET_DURATION = 120
YEAR_MIN = 1960
YEAR_MAX = 2022

def load_track():
    
    signal, sr = librosa.load(INPUT_FILE, sr=SAMPLE_RATE)
    print(signal.shape)
    
    duration = librosa.get_duration(y=signal, sr=sr)
    print(duration)

    if duration < TARGET_DURATION:
        return;

    samples_per_track = SAMPLE_RATE * TARGET_DURATION
    first_sample = int((len(signal) - samples_per_track) / 2) 
    start = first_sample
    finish = first_sample + samples_per_track
    mfccs = librosa.feature.mfcc(y=signal[start:finish], sr=sr, n_mfcc=NUM_MFCC, n_fft=NUM_FFT, hop_length=HOP_LENGTH)
    mfccs = mfccs.T
    
    outputs = {
        "labels": [],
        "mfccs": []
    }
       
    outputs["mfccs"].append(mfccs.tolist())
            
    with open(OUTPUT_FILE, "w") as fp:
        json.dump(outputs, fp, indent=4)

    return outputs

if __name__ == "__main__":
    example_data = load_track()
    examples = np.array(example_data["mfccs"])
    model = keras.models.load_model(MODEL_FILE)
       
    predictions = model.predict(examples)[0]     
    sum = 0

    for i, weight in enumerate(predictions):
        sum += i * weight
    
    guessed_year = round(sum + YEAR_MIN)
    print(guessed_year)

