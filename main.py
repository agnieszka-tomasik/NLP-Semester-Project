import nltk
import random
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tag.perceptron import PerceptronTagger
import librosa
import scipy.stats as stats
import glob as dir_iterator
from afinn import Afinn
import librosa.display
import numpy as np
import re
import matplotlib.pyplot as plt
#import SpeechRecognition
import speech_recognition as sr
#https://realpython.com/python-speech-recognition/

class FearClassifier:

    classifier = nltk.NaiveBayesClassifier
    fear_words = []

    def __init__(self):

        ps = PorterStemmer()
        size = 0
        fear = open('fear.txt', 'r').read().splitlines()
        for word in fear:
            stem = ps.stem(word.lower())
            if stem not in self.fear_words:
                self.fear_words.append(stem)
                size += 1
        print('SIZE OF FEAR WORDS IN LIST: ' + str(size))

    def find_features(self, line: str) -> dict:

        tokens = nltk.word_tokenize(line)
        words = set(tokens)
        features = {}
        for w in words:
            if w.lower() in self.fear_words:
                features[w.lower()] = 'fear'
            else:
                features[w.lower()] = 'neutral'
        return features

    def amp_audio(self, file: str):

        data, sr = librosa.core.load(file, sr=22050, mono=True, offset=1.2, duration=None)
        # data : type = array = audio time series
        amplitude = str(data.shape)

        print("Amplitude: ", amplitude.replace('(', '').replace(',', '').replace(')', ''))
        final_amp = amplitude.replace('(', '').replace(',', '').replace(')', '')
        print("Sampling rate: ", sr)

        # Sound amplitude plot
        time = np.arange(0, len(data)) / sr
        fig, ax = plt.subplots()
        ax.plot(time, data)
        ax.set(xlabel='Time(s)', ylabel='sound amplitude')
        plt.show()

        return final_amp

    def freq_audio(self, file):

        data, sr = librosa.core.load(file, sr=22050, mono=True, offset=1.2, duration=None)
        #spectral centroid of sound wave (inflection point) - frequency
        # The spectral centroid is commonly associated with the measure of the brightness of a sound.
        # https://librosa.org/doc/0.8.0/generated/librosa.feature.spectral_centroid.html
        FRAME_SIZE = 1024
        HOP_LENGTH = 512

        cent = librosa.feature.spectral_centroid(y=data, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        sc = str(cent.shape)
        final_sc = sc.replace('(', '').replace(',', '').replace(')', '')
        print("Spectral centroid: ", final_sc)

        S, phase = librosa.magphase(librosa.stft(y=data))
        times = librosa.times_like(cent)
        fig, ax = plt.subplots()
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
        ax.plot(times, cent.T, label='Spectral centroid', color='w')
        ax.legend(loc='upper right')
        ax.set(title='log Power spectrogram')
        plt.show()

        return final_sc

    def pitch_audio(self, file):

        data, sr = librosa.core.load(file, sr=22050, mono=True, offset=1.2, duration=None)
        pitches, magnitudes = librosa.piptrack(y=data, sr=sr, fmin=5, fmax=1600)
        pitch_ = str(pitches.shape)
        # since pitch_ is (frequency, time) use regex to just get string of frequency
        regex_pitch = re.sub(', [0-9]', '', pitch_)
        """
        ``pitches[f, t]`` contains instantaneous frequency at bin
        ``f``, time ``t``
        ``magnitudes[f, t]`` contains the corresponding magnitudes.
        Both ``pitches`` and ``magnitudes`` take value 0 at bins
        of non-maximal magnitude.
        """
        final_pitch = regex_pitch.replace('(', '').replace(',', '').replace(')', '')

        # Sound amplitude plot
        time = np.arange(0, len(data)) / sr
        fig, ax = plt.subplots()
        ax.plot(time, data)
        ax.set(xlabel='Time(s)', ylabel='sound pitch')
        plt.show()

        return final_pitch


    def speech_to_text(self, file: str):
        r = sr.Recognizer()
        # r.recognize_google() #API from Google
        audio = sr.AudioFile(file)
        with audio as source:
            audio = r.record(source)
        # need to go from audioFile to audioData
        print(r.recognize_google(audio))

    def find_POS(self, lines: list) -> list:  # I think we can use POS to see trends in neutral vs fear sentences, I + verb words typically in fear
        # using lines from sentence examples, do pos before removing words
        tokens = []
        tagger = PerceptronTagger()
        tagger.train([[('today', 'NN'), ('is', 'VBZ'), ('good', 'JJ'), ('day', 'NN')],
                      [('yes', 'NNS'), ('it', 'PRP'), ('beautiful', 'JJ')]])
        for sent in lines:
            tokens.append(nltk.word_tokenize(sent))
        for word in tokens:
            words = list(word)
            tagger.tag(words)

    def remove_stop_words(self):  # clears up space in database and improves processing time

        stopwords = set(nltk.corpus.stopwords.words('english'))
        stopwords.remove("the")
        stopwords.remove("a")
        stopwords.remove("an")
        stopwords.remove("in")
        stopwords.remove("but")
        stopwords.remove("by")

    def train(self, fear_lines: list, neutral_lines: list):

        documents = ([(line, 'fear') for line in fear_lines] + [(line, 'neutral') for line in neutral_lines])
        random.shuffle(documents)

        feature_sets = [(self.find_features(line), genre) for (line, genre) in documents]
        size = int(len(feature_sets)*0.1)
        train_set, test_set = feature_sets[size:], feature_sets[:size]

        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
        print('Classifier accuracy percent:', (nltk.classify.accuracy(self.classifier, test_set)) * 100)
        self.classifier.show_most_informative_features(100)

    def meter(self, lines: list):

        # currently, the meter is solely based off of
        # whether there are more fear words than neutral words;
        # not set in stone, may need to change, because
        # I could see that leading to a lot misinterpretations

        for line in lines:
            genre = self.classifier.classify(self.find_features(line))
            if genre == 'fear':
                print('FEAR: ' + line)
            else:
                print('NEUTRAL: ' + line)


    """
    Function compares amplitudes between an audio file that expresses fear versus a neutral audio file.
    Creates an average of amplitude values, if average of the fear amplitude is lower than the average of the neutral amplitude 
    than we can conclude that lower amplitude is a good indicator of fear.
    """

    def compare_amplitudes(self, files1, files2):

        list_amplitudes_fear = []
        list_amplitudes_neutral = []
        fear_total = 0
        neutral_total = 0

        # store amplitude values in a list
        for file in files1:
            list_amplitudes_fear.append(int(self.amp_audio(file)))
        for file in files2:
            list_amplitudes_neutral.append(int(self.amp_audio(file)))

        # get total amplitudes
        for amplitude in list_amplitudes_fear:
            fear_total += amplitude

        for amplitude in list_amplitudes_neutral:
            neutral_total += amplitude

        # get averages
        fear_avg = int(fear_total / len(list_amplitudes_fear))
        neutral_avg = int(neutral_total/ len(list_amplitudes_neutral))


        # tabulate data via Panda and print to visualize table
        df_fear = pd.DataFrame(list_amplitudes_fear, columns=['Fear'])
        df_neutral = pd.DataFrame(list_amplitudes_neutral, columns=['Neutral'])
        df = pd.concat([df_fear, df_neutral], axis=1)
        print(df)

        fvalue, pvalue = stats.f_oneway(df_fear, df_neutral)
        print("ANOVA statistical analysis F-value: ", fvalue)
        print("ANOVA statistical analysis p-value: ", pvalue)

        print("Fear Avg Amplitude: ", fear_avg)
        print("Neutral Avg Amplitude: ", neutral_avg)
        print("")

        # compare results under a 5% significance level
        if fear_avg < neutral_avg:
            print("Fear is proven to have a smaller amplitude based on the averages as fear:", fear_avg, "< neutral:", neutral_avg)
            return True
        else:
            print("Wrong")
            return False


    def compare_inflection(self, files1, files2):

        list_freq_fear = []
        list_freq_neutral = []
        fear_total = 0
        neutral_total = 0

        # store freq values in a list
        for file in files1:
            list_freq_fear.append(int(self.freq_audio(file)))
        for file in files2:
            list_freq_neutral.append(int(self.freq_audio(file)))

        # get total freq
        for freq in list_freq_fear:
            fear_total += freq

        for freq in list_freq_neutral:
            neutral_total += freq

        # get averages
        fear_avg = int(fear_total / len(list_freq_fear))
        neutral_avg = int(neutral_total/ len(list_freq_neutral))

        # tabulate data via Panda and print to visualize table
        df_fear = pd.DataFrame(list_freq_fear, columns=['Fear'])
        df_neutral = pd.DataFrame(list_freq_neutral, columns=['Neutral'])
        df = pd.concat([df_fear, df_neutral], axis=1)
        print(df)

        fvalue, pvalue = stats.f_oneway(df_fear, df_neutral)
        print("ANOVA statistical analysis F-value: ", fvalue)
        print("ANOVA statistical analysis p-value: ", pvalue)

        print("Fear Avg Freq: ", fear_avg)
        print("Neutral Avg Freq: ", neutral_avg)
        print("")

        # compare results under a 5% significance level
        if fear_avg < neutral_avg:
            print("Fear is proven to have a smaller frequency based on the averages as fear:", fear_avg, "< neutral:", neutral_avg, "\n")
            return True
        else:
            print("Wrong")
            return False

    def compare_pitch(self, files1, files2):

        list_pitch_fear = []
        list_pitch_neutral = []
        fear_total = 0
        neutral_total = 0

        # store amplitude values in a list
        for file in files1:
            print("pitch:", self.pitch_audio(file))
            list_pitch_fear.append(int(self.pitch_audio(file)))
        for file in files2:
            list_pitch_neutral.append(int(self.pitch_audio(file)))

        # get total amplitudes
        for freq in list_pitch_fear:
            fear_total += freq

        for freq in list_pitch_neutral:
            neutral_total += freq

        # get averages
        fear_avg = int(fear_total / len(list_pitch_fear))
        neutral_avg = int(neutral_total / len(list_pitch_neutral))

        # tabulate data via Panda and print to visualize table
        df_fear = pd.DataFrame(list_pitch_fear, columns=['Fear'])
        df_neutral = pd.DataFrame(list_pitch_neutral, columns=['Neutral'])
        df = pd.concat([df_fear, df_neutral], axis=1)
        print(df)

        fvalue, pvalue = stats.f_oneway(df_fear, df_neutral)
        print("ANOVA statistical analysis F-value: ", fvalue)
        print("ANOVA statistical analysis p-value: ", pvalue)

        print("Fear Avg Pitch: ", fear_avg)
        print("Neutral Avg Pitch: ", neutral_avg)
        print("")

        # compare results under a 5% significance level
        if fear_avg > neutral_avg:
            print("Fear is proven to have a larger pitch based on the averages as fear:", fear_avg, "> neutral:",
                  neutral_avg, "\n")
            print("Pitch is based off the instantenous frequency")
            return True
        else:
            print("Wrong")
            return False


    """
    Analyze the context of words to determine if they convey fear (use POS tags here perhaps a tree)?
    """
    def word_context(self):

        return "context"


    """
    Check if the lemmatization of the text provides evidence of fear
    """

    def lemmatization(self):

        return None

    # need to compare audio results and text results, unsure of how to do this currently, use classifier on both?
    def audio_vs_text(self):
        return None

    def evaluate(self):
        return None

    def sentiment_score(self,file):
        af = Afinn()

        # compute sentiment scores (polarity) and labels
        sentiment_scores = [af.score(file)]
        sentiment_category = ['positive' if score > 0
                              else 'negative' if score < 0
                                else 'neutral'
                                 for score in sentiment_scores]
        print(sentiment_scores)
        print(sentiment_category)


if __name__ == '__main__':
    fear = open('fear_example.txt', 'r').read().splitlines()
    neutral = open('neutral_example.txt', 'r').read().splitlines()
    test = open('test.txt', 'r').read().splitlines()
    fear_audio_files = dir_iterator.glob("./Fear Audio Files/*.wav")
    neutral_audio_files = dir_iterator.glob("./Neutral Audio Files/*.wav")

    fc = FearClassifier()
    fc.train(fear, neutral)
    fc.meter(test)
    fc.find_POS(test)
    print("Fear audio example: ")
    fc.amp_audio('Fear Audio Files/OAF_pain_fear.wav')
    print("Neutral audio example: ")
    fc.amp_audio('Neutral Audio Files/OAF_pain_neutral.wav')
    fc.speech_to_text('Fear Audio Files/OAF_pain_fear.wav')
    fc.compare_amplitudes(fear_audio_files, neutral_audio_files)
    fc.compare_inflection(fear_audio_files,neutral_audio_files)
    fc.compare_pitch(fear_audio_files, neutral_audio_files)
    fc.sentiment_score("fear text: ",open('fear_example.txt', 'r').read())
    fc.sentiment_score("neutral text: ", open('neutral_example.txt', 'r').read())