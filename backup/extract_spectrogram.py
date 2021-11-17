import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

'''
Read wav files from SOURCE folder, extract spectrograms in JPG format, and save in TARGET folder
'''


class SpectrogramExtractor:
    def extract(self, SOURCE, TARGET, FIG_SIZE):
        os.chdir(SOURCE)
        for file in os.listdir(SOURCE):
            # load audio file with Librosa
            signal, sample_rate = librosa.load(file, sr=22050)

            # perform Fourier transform (FFT -> power spectrum)
            fft = np.fft.fft(signal)

            # calculate abs values on complex numbers to get magnitude
            spectrum = np.abs(fft)

            # create frequency variable
            f = np.linspace(0, sample_rate, len(spectrum))

            # take half of the spectrum and frequency
            left_spectrum = spectrum[:int(len(spectrum)/2)]
            left_f = f[:int(len(spectrum)/2)]

            # STFT -> spectrogram
            hop_length = 512  # in num. of samples
            n_fft = 2048  # window in num. of samples

            # calculate duration hop length and window in seconds
            hop_length_duration = float(hop_length)/sample_rate
            n_fft_duration = float(n_fft)/sample_rate

            # perform stft
            stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

            # calculate abs values on complex numbers to get magnitude
            spectrogram = np.abs(stft) ** 2

            # apply logarithm to cast amplitude to Decibels
            log_spectrogram = librosa.amplitude_to_db(spectrogram)

            # Matplotlib plots: removing axis, legends and white spaces
            plt.figure(figsize=FIG_SIZE)
            plt.axis('off')
            librosa.display.specshow(
                log_spectrogram, sr=sample_rate, hop_length=hop_length)
            plt.savefig(f'{TARGET}\\{file[0:-4]}.jpg',
                        bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":

    '''
    Loading audio files with Librosa
    '''
    # SOURCE = "C:/data/audio"
    SOURCE = "C:/data/in"
    # SOURCE = "C:/data/baseline"
    # SOURCE = "C:/data/rattle"
    # TARGET = "C:/data/spectrogram/baseline"
    TARGET = "C:/data/out"
    # TARGET = "C:/data/spectrogram/rattle"
    FIG_SIZE = (40, 40)
    # FIG_SIZE = (50, 20)

    # instantiate all objects
    extractor = SpectrogramExtractor()
    extractor.extract(SOURCE, TARGET)
