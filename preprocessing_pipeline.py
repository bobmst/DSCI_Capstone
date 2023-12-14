from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import librosa
import noisereduce as nr

def reduce_noise(signal, sr):
    """
    Reduce noise from spectrogram
    """
    noise_reduced = nr.reduce_noise(signal, sr=sr)
    return noise_reduced

def pad_trim_audio(signal, sr, length=15):
    """
    Pad or trim audio signal to a fixed length
    """
    target_length = sr * length
    if len(signal) > target_length:
        clean_signal = signal[:target_length]
    elif len(signal) < target_length:
        pad_width = target_length - len(signal)
        clean_signal = np.pad(signal, (0, pad_width), 'constant')
    return clean_signal

# may need to play around with these parameters
def compute_spectrogram(signal, n_fft=2048):
    """
    Compute spectrogram of audio signal
    win_length: window length in samples, smaller values for better temporal resolution. Default is win_length=n_fft
    hop_length: number of samples between successive frames, smaller values increase number of columns D . Default
    is hop_length=win_length//4
    """
    spectrogram = librosa.stft(signal, n_fft=n_fft)
    spectrogram = np.abs(spectrogram)
    return spectrogram

def convert_to_decibel(spectrogram):
    """
    Convert spectrogram to decibel
    """
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return spectrogram

def compute_melspectrogram(signal, n_mels=128):
    """
    Extract mel-spectrogram from audio signal
    """
    spectrogram = librosa.feature.melspectrogram(S=signal, n_mels=n_mels)
    return spectrogram

def compute_mfcc(melspectrogram, n_mfcc=20):
    """
    Extract MFCC from Mel-spectrogram
    """
    spectrogram = librosa.feature.mfcc(S=melspectrogram, n_mfcc=n_mfcc)
    return spectrogram

def process_audio(signal, sr, length=15, n_fft=2048, n_mels=128, n_mfcc=20):
    """
    Process audio signal
    """
    noise_reducded = reduce_noise(signal, sr)
    clean_signal = pad_trim_audio(noise_reducded, sr, length=length)
    spectrogram = compute_spectrogram(clean_signal, n_fft=n_fft)
    spectrogram = convert_to_decibel(spectrogram)
    melspectrogram = compute_melspectrogram(spectrogram, n_mels=n_mels)
    mfcc = compute_mfcc(melspectrogram, n_mfcc=n_mfcc)
    return (spectrogram, melspectrogram, mfcc)

##TODO (decide if sckit-learn is the framework)
class NoiseReducer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = []
        for signal, sr in X:
            noise_reduced = reduce_noise(signal, sr=sr)
            data.append((noise_reduced, sr))

        return {'noise_reduced': data}


class AudioLengthNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, length=15):
        self.length = length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        length_normalized = np.array([pad_trim_audio(signal, sr) for signal, sr in X])
        return {'length_normalized': length_normalized}

    def pad_trim_audio(self, signal, sr):
        target_length = sr * self.length
        if len(signal) > target_length:
            return signal[:target_length]
        elif len(signal) < target_length:
            pad_width = target_length - len(signal)
            return np.pad(signal, (0, pad_width), 'constant')
        else:
            return signal

class SpectrogramExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_fft=2048):
        self.n_fft = n_fft

    def fit(self, X, y=None):
        return self

    def compute_spectrogram(self, signal):
        spectrogram = librosa.stft(signal, n_fft=self.n_fft)
        return np.abs(spectrogram)

    def convert_to_decibel(self, spectrogram):
        return librosa.amplitude_to_db(spectrogram, ref=np.max)

    def transform(self, X):
        return np.array([self.convert_to_decibel(self.compute_spectrogram(signal)) for signal in X])

class MelSpectrogramExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_mels=128):
        self.n_mels = n_mels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([librosa.feature.melspectrogram(S=signal, n_mels=self.n_mels) for signal in X])

class MFCCExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_mfcc=20):
        self.n_mfcc = n_mfcc

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([librosa.feature.mfcc(S=signal, n_mfcc=self.n_mfcc) for signal in X])

