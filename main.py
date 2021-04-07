import nltk
import random
from nltk.stem import PorterStemmer


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
            fear = False
            features = self.find_features(line)
            for (_, genre) in features.items():
                if genre == 'fear':
                    fear = True
            if fear:
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