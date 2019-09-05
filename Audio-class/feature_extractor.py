import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import soundfile as sf

##Return audio features 
def feature_extraction(file_name):
    X, sample_rate = librosa.load(file_name)
    if X.ndim > 1:
        X = X[:,0]
    X = X.T
    
    # Get features   
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    print(mfccs.shape)#40 values
    zcr = np.mean(librosa.feature.zero_crossing_rate(X))
    print(zcr.shape)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    print(chroma.shape)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    print(mel.shape)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    print(contrast.shape)
    rolloff =  np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sample_rate).T, axis=0)
    print(rolloff.shape)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    print(tonnetz.shape)#tonal centroid features
    
    return mfccs, chroma, mel, contrast, tonnetz,rolloff,zcr
    

def parse_audio_files(parent_dir, sub_dirs, file_ext='*.wav'): 
    features, labels = np.empty((0,195)), np.empty(0) # 193 features total. This can vary
    
    for label, sub_dir in enumerate(sub_dirs): ##The enumerate() function adds a counter to an iterable.
        for file_name in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)): ##parent is data, sub_dirs are the classes
            try:
                mfccs, chroma, mel, contrast, tonnetz ,rolloff,zcr= feature_extraction(file_name)
                
            except Exception as e:
                print("[Error] there was an error in feature extraction. %s" % (e))
                continue
                
            extracted_features = np.hstack([mfccs,chroma, mel, contrast, tonnetz,rolloff,zcr]) #Stack arrays in sequence horizontally (column wise)
            features = np.vstack([features, extracted_features]) #Stack arrays in sequence vertically (row wise).
            labels = np.append(labels, label)
        print("Extracted features from %s, done" % (sub_dir))
    return np.array(features), np.array(labels, dtype = np.int) ## arrays with features and corresponding labels for each audio

def one_hot_encode(labels): ##Check this hot encode
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

audio_directories = os.listdir(r"D:\Compressed\ESC-50-master\ESC-50-master\Data")
audio_directories.sort()
features, labels = parse_audio_files(r"D:\Compressed\ESC-50-master\ESC-50-master\Data", audio_directories) #(parent dir,sub dirs)
np.save('feat.npy', features) 
np.save('label.npy', labels)
labels = np.load('label.npy')

