import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import butter, sosfilt
from matplotlib import pyplot as plt
import os

# Define a function to apply a bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def iterate_files():
    directory = 'train'
    iterator = 0
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            analysis(os.path.join(directory, filename), iterator)
            iterator += 1

def analysis(path, iterator=0):
    # Load the .wav file
    sample_rate, data = wavfile.read(path)

    # If the audio file has more than one channel (i.e., it's a stereo file), convert it to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Apply a bandpass filter to the audio data
    data = butter_bandpass_filter(data, 80, 255, sample_rate)

    # Normalize the audio data
    data = data / np.max(np.abs(data))

    # Apply FFT
    fft_out = fft(data)

    # Calculate the absolute value of each FFT point
    abs_fft = np.abs(fft_out)

    # Only consider the first half of the frequencies and FFT output
    abs_fft = abs_fft[:len(abs_fft)//2]
    frequencies = np.fft.fftfreq(len(abs_fft)*2, 1.0/sample_rate)[:len(abs_fft)]

    # Find the peak frequency in the product spectrum
    peak_frequency = abs(frequencies[np.argmax(abs_fft)])

    # Classify based on the peak frequency
    if 85 <= peak_frequency <= 180:
        #print(peak_frequency)
        print(iterator+1, "M")
    elif 165 <= peak_frequency <= 255:
        #print(peak_frequency)
        print(iterator+1, "K")
    else:
        #print(peak_frequency)
        print(iterator+1, "U")

def main():
    iterate_files()

# # Plot the spectrum
# plt.figure(figsize=(14, 5))
# plt.plot(frequencies, abs_fft)
# #plt.plot(frequencies[:len(product)], product)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.title('Spectrum')
# plt.xlim(0, 500)
# plt.show()

if __name__ == '__main__':
    main()