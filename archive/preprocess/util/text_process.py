import pickle

from nltk.tokenize import  sent_tokenize,word_tokenize,TweetTokenizer
from nltk.corpus import stopwords
# from archive.preprocess.util.porter_stemmer import PorterStemmer
import nltk
import os
import re


class TextProcess:
    dictionary = None
    UNK = "<UNK>"

    tag_filter = ["NOUN","ADJ","ADP"]

    tknzr = TweetTokenizer()
    # stemmer = ()

    interested_words = set()

    stem_pairs = set()
    stopword = set(stopwords.words('english'))

    def __init__(self, dictionary):
        self.dictionary = dictionary

    @classmethod
    def initiliaze(cls, path_pretraining):
        # stem the pretraining file

        if path_pretraining is not None:

            if os.path.exists(path_pretraining+'.cache'):
                with open(path_pretraining+'.cache', 'rb') as f:
                    TextProcess.interested_words = pickle.load(f)
                    print(TextProcess.interested_words)

            else:
                f_pretraining = open(path_pretraining, "r")
                for line in f_pretraining:
                    items = line.split(" ")
                    # print(items)
                    word = items[0]
                    # print(word)
                    TextProcess.interested_words.add(word)
                with open(path_pretraining+'.cache', 'wb') as f:
                    pickle.dump(TextProcess.interested_words, f)
        return

    @classmethod
    def process_word(cls, word, tag, stem_flag = True, validate_flag = True, pos_filter = True, remove_stop_word = False, only_interested_words = False):
        '''
        filter some words based on some heuristics
        :param word:
        :param tag:
        :param stem_flag:
        :param validate_flag:
        :param pos_filter:
        :param remove_stop_word:
        :param only_interested_words:
        :return:
        '''
        # if pos_filter and tag not in TextProcess.tag_filter:
        #     return TextProcess.UNK
        if remove_stop_word and word in stopwords.words('english'):
            return TextProcess.UNK
        if validate_flag:
            if len(word) <= 2 or len(word) >= 16:
                return TextProcess.UNK
            # for ch in word:
            #     if not ch.isalpha():
            #         return TextProcess.UNK

        # if stem_flag:
        #     origin_word = word
        #     word = stemmer.stem(word,0,len(word)-1)
        #     stem_pair = (origin_word, word)
        #     TextProcess.stem_pairs.add(stem_pair)
        # if word not in TextProcess.interested_words:
        #     return TextProcess.UNK
        return word

    @classmethod
    def generateStemPair(cls):
        home = os.environ["HOME"]
        path_stempairs = "".join((home, "/data/yelp/stem_pairs.txt"))
        f_stempairs = open(path_stempairs, "w")

        batch = ""
        for oword, nword in TextProcess.stem_pairs:
            line = "\t".join([oword, nword])
            line = "".join([line, "\n"])
            batch = "".join([batch, line])
            if len(batch) >= 1000000:
                f_stempairs.write(batch)
                batch = ""
        f_stempairs.write(batch)


    def process(self, text):
        text = text.lower()
        text_processed = ""
        '''
        remove trivial words (stopwords etc.)
        '''
        for word in re.split("\W+", text):

            if word in self.stopword:
                continue
            if len(word) <= 2 or len(word) >= 15:
                continue
            if self.dictionary[word] <= 8 or self.dictionary[word] >= 10000:
                continue
            if word not in TextProcess.interested_words:
                continue
            text_processed += word + ' '

        return text_processed


# if __name__ == '__main__':
#     import os
#     home = os.environ["HOME"]
#     path_pretraining = "".join((home, "/Data/glove/glove.6B.200d.txt"))
#     textprocess = TextProcess.initiliaze(path_pretraining)
#     print("init")
#     print(textprocess.process(text="fequency statics i eat djias but i do not like chicken"))