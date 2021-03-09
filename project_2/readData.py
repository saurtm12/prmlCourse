import librosa as lb
import numpy as np
import csv

import  scipy
def readData():
    X_train = []
    X_test = []
    y_train = []
    #create filter to remove low frequencies, as bird fundamental frequencies are often between 500-2000, and their harmonic is between 500 and 8000
    b, a = scipy.signal.butter(5, [500*2/44100, 8000*2/44100], btype='band')
    
    with open("ff1010bird_metadata_2018.csv") as f:
        data = csv.reader(f)
        next(data, None)

        for row in data:
            name, dataset, label = row

            audio, fs = lb.load(f"{dataset}_wav/wav/{name}.wav", sr=None)
            dur = lb.get_duration(audio, sr=fs, n_fft=1024, hop_length=512)
            if dur > 10:
                continue
            elif dur < 10:
                audio = lb.util.fix_length(audio, 441000)
            
            #filtering out the low frequencies
            audio = scipy.signal.lfilter(b,a,audio)
            #preamplify the signal
            audio = scipy.signal.lfilter([1, -0.95], [1], audio)
            audio = audio / np.max(np.abs(audio))

            kwargs_for_mel = {"n_mels": 40}
            x = lb.feature.melspectrogram(
                y=audio, 
                sr=44100, 
                n_fft=1024, 
                hop_length=512, 
                **kwargs_for_mel)
            
            #extract logmel spectogram
            x = np.log10(x.T)
            X_train.append(x)
            y_train.append(label)

    with open("warblrb10k_public_metadata_2018.csv") as f:
        data = csv.reader(f)
        next(data, None)

        for row in data:
            name, dataset, label = row

            audio, fs = lb.load(f"{dataset}_public_wav/wav/{name}.wav", sr=None)
            dur = lb.get_duration(audio, sr=fs, n_fft=1024, hop_length=512)
            if dur > 10:
                continue
            elif dur < 10:
                audio = lb.util.fix_length(audio, 441000)
            #filtering out the low frequencies
            audio = scipy.signal.lfilter(b,a,audio)
            #preamplify the signal
            audio = scipy.signal.lfilter([1, -0.95], [1], audio)
            audio = audio * 1 / np.max(np.abs(audio))

            kwargs_for_mel = {"n_mels": 40}
            x = lb.feature.melspectrogram(
                y=audio, 
                sr=44100, 
                n_fft=1024, 
                hop_length=512, 
                **kwargs_for_mel)
            x = x.T
            x = np.log10(x.T)
            X_train.append(x)
            y_train.append(label)
            
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    for name in range(4512):
        audio = np.load(f"audio/{name}.npy")
        audio = lb.resample(audio, 48000, 44100)
        audio = scipy.signal.lfilter(b,a,audio)
        audio = scipy.signal.lfilter([1, -0.95], [1], audio)
        audio = audio * 1 / np.max(np.abs(audio))

        kwargs_for_mel = {"n_mels": 40}
        x = lb.feature.melspectrogram(
            y=audio,
            sr=44100,
            n_fft=1024,
            hop_length=512,
            **kwargs_for_mel)
        #extract logmel spectogram
        x = np.log10(x.T)

        X_test.append(x)


    np.save("X_test.npy", X_test)
