import numpy as np
import warnings
from scipy.io import wavfile
from scipy.fftpack import fft
import sys

# def iterate_files():
#     ans = []
#     directory = 'wav'
#     iterator = 0
#     for filename in os.listdir(directory):
#         if filename.endswith('.wav'):
#             ans.append(analysis(os.path.join(directory, filename), iterator))
#             iterator += 1
#     return ans

def analysis(path, iterator=0):
    # Ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Load the .wav file
        sample_rate, data = wavfile.read(path)

    # If the audio file has more than one channel (i.e., it's a stereo file), convert it to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Normalize the audio data
    data = data / np.max(np.abs(data))

    # Divide the audio data into chunks
    chunk_size = round(sample_rate)
    num_chunks = len(data) // chunk_size

    # Classify each chunk
    male_chunks = 0
    female_chunks = 0
    for i in range(num_chunks):
        chunk = data[i*chunk_size:(i+1)*chunk_size]

        # Apply FFT
        fft_out = fft(chunk)

        # Calculate the absolute value of each FFT point
        abs_fft = np.abs(fft_out)

        # Only consider the first half of the frequencies and FFT output
        abs_fft = abs_fft[:len(abs_fft)//2]
        frequencies = np.fft.fftfreq(len(abs_fft)*2, 1.0/sample_rate)[:len(abs_fft)]

        # Create a mask for frequencies
        mask = (85 <= frequencies) & (frequencies <= 255)

        # Apply the mask to the frequencies and FFT output
        frequencies = frequencies[mask]
        abs_fft = abs_fft[mask]

        # Find the peak frequency in the product spectrum
        peak_frequency = abs(frequencies[np.argmax(abs_fft)])

        # Classify based on the peak frequency
        if 85 <= peak_frequency < 165:
            male_chunks += 1
        elif 165 <= peak_frequency <= 255:
            female_chunks += 1

    # Classify the whole track
    if male_chunks > female_chunks:
        return "M"
    else:
        return "K"

# def get_answer_from_name():
#     names = []
#     for filename in os.listdir('wav'):
#         if filename.endswith('.wav'):
#             if filename[4] == 'M':
#                 names.append('M')
#             else:
#                 names.append('K')
#     return names

# def get_accuracy(answers, names):
#     count = 0
#     for i in range(len(answers)):
#         if answers[i] == names[i]:
#             count += 1
#     return count / len(answers)

def main():
    #answers = iterate_files()
    # names = get_answer_from_name()
    # print(get_accuracy(answers, names))
    filename = sys.argv[1]

    print(analysis(filename))




if __name__ == '__main__':
    main()