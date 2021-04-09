import nltk
import random
from nltk.stem import PorterStemmer
from nltk.tag.perceptron import PerceptronTagger
import librosa
import numpy as np
import matplotlib.pyplot as plt
#import SpeechRecognition
import speech_recognition as sr
#https://realpython.com/python-speech-recognition/

class FearClassifier:
    # for later development: if we save the classifier to a file with pickle,
    # we'll be able to reload it and see the trend of performance each time it is trained
    # since that information won't be lost at the end of the program but rather stored
    # somewhere

    classifier = nltk.NaiveBayesClassifier
    fear_words = []

    def __init__(self):
        # fear.txt will be a file containing all of the popular
        # words that denote fear that we have compiled and saved

        # NOTE: may be more effective to use stemmer to create
        # bag of words that word_features will be constructed from
        # e.g. 'apprehensiveness' will register fear since it's
        # stem is 'apprehensive', which will be included in word_features

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

    def audio(self, file: str):

        data, sr = librosa.core.load(file, sr=22050, mono=True, offset=1.2, duration=None)
        # data : type = array = audio time series
        print(data.shape)
        print("sampling rate: ", sr)

        time = np.arange(0, len(data)) / sr
        fig, ax = plt.subplots()
        ax.plot(time, data)
        ax.set(xlabel='Time(s)', ylabel='sound amplitude')
        plt.show()

    def speech_to_text(self, file: str):
        r = sr.Recognizer()
        # r.recognize_google() #API from Google
        audio = sr.AudioFile(file)
        print(type(audio))
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


if __name__ == '__main__':
    fear = open('fear_example.txt', 'r').read().splitlines()
    neutral = open('neutral_example.txt', 'r').read().splitlines()
    test = open('test.txt', 'r').read().splitlines()

    fc = FearClassifier()
    fc.train(fear, neutral)
    fc.meter(test)
    fc.find_POS(test)
    print("Pain audio example: ")
    fc.audio('Fear Audio Files/OAF_pain_fear.wav')
    print("Neutral audio example: ")
    fc.audio('Neutral Audio Files/OAF_pain_neutral.wav')
    # fc.speech_to_text('Fear Audio Files/OAF_pain_fear.wav')
