# encoding: utf-8
from __future__ import generators
from collections import Counter
import pickle
import os
import spacy
import re
import csv
import marisa_trie
import nltk

from nltk.stem.porter import PorterStemmer
import heapq
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import gensim
from gensim import corpora
from gensim.matutils import sparse2full
from collections import defaultdict

nlp = spacy.load('en')
IS_STEM=True
REMOVE_STOPWORDS=True
stemmer = PorterStemmer()

def stem(word):
    stemword = word.strip()
    if(len(stemword) > 3):
        stemword = stemmer.stem(stemword)

    return stemword

def multiwordstem(word_list ):
    for i in range(len(word_list)):
        word_list[i] = stem(word_list[i])
    return ' '.join(word_list)

def preprocessToken(text):
    return re.sub(r'\W+|\d+', '', text.strip().lower())

def preprocessText(text,stemming=True,stopwords_removal=True):
    # print(text)
    text = re.sub("[ ]{1,}",r' ',text)
    text = re.sub(r'\W+|\d+', ' ', text.strip().lower())
    tokens = [token.strip()  for token in text.split(" ")]
    tokens = [token for token in tokens if len(token) > 1]
    if stopwords_removal:
        tokens = [token for token in tokens if token not in stopwords]
    if stemming:
        tokens = [stem(token) for token in tokens ]

    tokens = [token.strip() for token in tokens if len(token.strip()) > 1]
    return tokens

class KeywordList:
    def __init__(self, name):
        self.name = name
        self.wordlist = WORDLIST_PATH[name]
        print (name)
        self.triepath = TRIE_CACHE_DIR+name+'_trie_dict.cache'
        self.trie = self.load_trie(self.triepath)

    def load_trie(self, trie_cache_file):
        '''
        Load a prebuilt tree from file or create a new one
        :return:
        '''
        trie = None

        if os.path.isfile(trie_cache_file):
            print('Start loading trie from %s' % trie_cache_file)
            with open(trie_cache_file, 'rb') as f:
                trie = pickle.load(f)
        else:
            print('Trie not found, creating %s' % trie_cache_file)
            count = 0
            listwords = []
            dict_files = [self.wordlist]
            for dict_file in dict_files:
                print(dict_file)
                file = open(dict_file, 'r')
                for line in file:
                    print(line)
                    tokens = preprocessText(line,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS)
                    if(len(tokens)>0):
                        listwords.append(tokens)

            trie = MyTrie(listwords)
            with open(trie_cache_file, 'wb') as f:
                pickle.dump(trie, f)
        return trie

def KnuthMorrisPratt(text, pattern):

    '''Yields all starting positions of copies of the pattern in the text.
Calling conventions are similar to string.find, but its arguments can be
lists or iterators, not just strings, it returns all matches, not just
the first one, and it does not need the whole text in memory at once.
Whenever it yields, it will have read the text exactly up to and including
the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos

__author__ = 'Memray'
'''
A self-implenmented trie keyword matcher
'''
#KEYWORD_LIST_PATH = '/home/memray/Project/acm/ACMParser/resource/data/phrases/'
KEYWORD_LIST_PATH = 'data/keyphrase/wordlist/'
ACL_KEYWORD_PATH = KEYWORD_LIST_PATH + 'acl_keywords.txt'
ACM_KEYWORD_PATH = KEYWORD_LIST_PATH + 'acm_keywords_168940.txt'
MICROSOFT_KEYWORD_PATH = KEYWORD_LIST_PATH + 'microsoft_keywords.txt'
WIKI_KEYWORD_PATH = KEYWORD_LIST_PATH + 'wikipedia_14778209.txt'
WIKI_LINKS_PATH = KEYWORD_LIST_PATH + 'link.txt'
WIKI_SECTION_PATH = KEYWORD_LIST_PATH + 'section.txt'
WIKI_ITALICS_PATH = KEYWORD_LIST_PATH + 'italics.txt'
WIKI_MERGEALL_PATH = KEYWORD_LIST_PATH + 'mergeall.final.txt'

WORDLIST_DIR = 'data/keyphrase/wordlist/'
TRIE_CACHE_DIR = 'data/keyphrase/extracted_keyword/'
WORDLIST_PATH = {'greedy-wiki':WORDLIST_DIR+'wikipedia_14778209.txt',
                 'greedy-acm':WORDLIST_DIR+'acm_keywords_168940.txt',
                 'wikilink': WORDLIST_DIR + 'link.txt',
                 'wikiitalic': WORDLIST_DIR + 'italics.txt',
                 'wikisection': WORDLIST_DIR + 'section.txt',
                 'wikimergeall': WORDLIST_DIR + 'mergeall.final.txt',
                'wikiacm': WORDLIST_DIR + 'wikiacm.txt',
                 'wikitemp':WORDLIST_DIR + 'wiki_123'
                 }
    #'greedy-wiki':WORDLIST_DIR+'wikipedia_14778209.txt',
dict_files = [ACM_KEYWORD_PATH]

OUTPUT_DIR      = '/Users/khushsi/Downloads/entity2vector-ircopy/data/keyphrase_output/'

KEYWORD_ONLY    = OUTPUT_DIR + 'keyword_only.txt'
WIKI_ONLY       = OUTPUT_DIR + 'wiki_only.txt'
TOP_TFIDF       = OUTPUT_DIR + 'tfidf_100.txt'
WIKI_links       = OUTPUT_DIR + 'wiki_only_list.txt'
WIKI_italics       = OUTPUT_DIR + 'wiki_only_italics.txt'
WIKI_section       = OUTPUT_DIR + 'wiki_only_section.txt'
WIKI_merge       = OUTPUT_DIR + 'wiki_only_merge.txt'




def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

import string

# def isEnglish(s):
#     return s.translate(None, string.punctuation).isalnum()
STOPWORD_PATH = 'data/stopword/stopword_en.txt'


def load_stopwords(sfile=STOPWORD_PATH):

    dict = set()
    file = open(sfile, 'r')
    for line in file:
        dict.add(line.lower().strip())
    return dict

stopwords = load_stopwords()

stopwords_min = load_stopwords('data/stopword/stopword_min.txt')

def isInChunk(word,chunklist):
    wordlist = word.split(" ")
    if word in chunklist:
        return True
    if len(wordlist) > 1:
        for chunk in chunklist:
            listchunk = chunk.split(" ")
            for s in KnuthMorrisPratt(listchunk,wordlist):
                return True
    return False

def isAlreadyPresent(word,presentlist):
    # print(presentlist)
    # print(word)
    for chunk in presentlist:
        listchunk = chunk[0].split(" ")
        for s in KnuthMorrisPratt(listchunk,word.split(" ")):
            # print(word)
            return True
    return False



class MyTrie:
    """
    Implement a static trie with  search, and startsWith methods.
    """
    def __init__(self,words):
        newlist = self.maketrie(words)
        self.nodes = marisa_trie.Trie(newlist)

    # Inserts a phrase into the trie.
    def maketrie(self, words):
        makelist = []
        for word  in words:
            current_word = ' '.join(word)
            makelist.append(current_word)
        return makelist

    # Returns if the word is in the trie.
    def search(self, words):
        if( words in self.nodes ):
            return True
        else:
            return False

    # Scan a sentence and find any ngram that appears in the sentence
    def scan(self, sentence, min_length=1, max_length=3):
        keyword_list = []
        tokens = preprocessText(sentence,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS)

        ngrams = []
        for i in range(min_length, max_length+1):
            ngrams += nltk.ngrams(tokens, i)

        for ngram in ngrams:
            if(self.search(' '.join(ngram))):
                keyword_list.append(' '.join(ngram))

        return keyword_list



class Document:
    def __init__(self, *args, **kwargs):
        self.sentences = []
        self.npchunks = []
        self.type = "general"
        self.id = args[0]
        self.text = args[1]
        self.type= args[0].split("-")[0]


        sen_list = sent_tokenize(self.text)

        for sen in sen_list:
            self.sentences.append(sen)
            # print(sen)
        self.no_sent = len(self.sentences)

    def __str__(self):
        return '%s\t%s' % (self.id, self.text)

IR_CORPUS = 'data/keyphrase/textbook/all_text.csv'


def load_document(path,booknames=['iir']):
    print('Start loading documents from %s' % path)
    doc_list = []
    file = open(path, 'r',encoding='utf-8', errors='ignore')

    with file as tsv:
        tsvin = csv.reader(file, delimiter=',')
        for row in tsvin:
            if row[0].startswith(tuple(booknames)):
                doc = Document(row[0].strip(),row[1].strip())
                doc_list.append(doc)
    return doc_list



def getGlobalngrams(grams,documents,threshold):

    singlecorpus = ""
    for doc in documents:
        singlecorpus += ' '+ doc.text + '\n'


    ncorpus = ' '.join(preprocessText(singlecorpus))
    tf = TfidfVectorizer(analyzer='word', ngram_range=grams, stop_words=stopwords)
    tfidf_matrix = tf.fit_transform([ncorpus])
    feature_names = tf.get_feature_names()
    doc = tfidf_matrix.todense()
    temptokens = zip(doc.tolist()[0], itertools.count())
    temptokens = [(x, y) for (x, y) in temptokens if x > threshold]
    tokindex = heapq.nlargest(len(temptokens), temptokens)
    global1grams = dict([(feature_names[y],x) for (x, y) in tokindex ])
    topindex = [ (feature_names[y],x)  for (x,y) in tokindex ]
    f = open('data/file'+str(grams[0])+".txt",'w')
    for key in global1grams:
        f.write(key+"\n")


    return  global1grams,topindex


def extract_np_high_tfidf_words( documents, top_k=200, ngram=(1,1), OUTPUT_FOL='TFIDF',is_global=True):
    '''
    Return the top K 1-gram terms according to TF-IDF
    Load corpus and convert to Dictionary and Corpus of gensim
    :param corpus_path
    :param num_feature, indicate how many terms you wanna retain, not useful now
    :return:
    '''

    if not os.path.exists(OUTPUT_DIR + OUTPUT_FOL):
        os.makedirs(OUTPUT_DIR + OUTPUT_FOL)

    texts = [[preprocessText(sen,stemming=False,stopwords_removal=False)  for sen in document.sentences] for document in documents]
    corpus = [' '.join(preprocessText(document.text)) for document in documents]

    npchunkcorpus = []
    npdocumentcorpus = {}

    iDoc = 0
    for text in texts:
        npdocumentcorpus[iDoc] = []
        for sen in text:
            ichunklist = list(nlp(' '.join(sen)).noun_chunks)
            npdocumentcorpus[iDoc] += ichunklist
            npchunkcorpus.append(ichunklist)
        iDoc += 1


    top_k_list = {}


    # chunkn=set()
    # for textnp in npchunkcorpus:
    #     for chunk in textnp:
    #         chunklisti = ' '.join([tok.lower() for tok in str(chunk).split(" ") if tok not in stopwords])

    for iDoc in npdocumentcorpus.keys():
        textnp = npdocumentcorpus[iDoc]
        chunklisti = []
        documents[iDoc].npchunks = []
        for chunk in textnp:
            chunklisti.append(' '.join(preprocessToken(str(chunk.text))))
        documents[iDoc].npchunks += chunklisti



    tf = TfidfVectorizer(analyzer='word', ngram_range=ngram,stop_words=stopwords,min_df=2)


    tfidf_matrix = tf.fit_transform(corpus)
    feature_names = tf.get_feature_names()

    doc_id=0

    for doc in tfidf_matrix.todense():
        temptokens = zip(doc.tolist()[0], itertools.count())
        temptokens1=[]
        for (x, y) in temptokens:
            stemy = feature_names[y]
            if x > 0.001:
                temptokens1.append((x,y))

        tokindex = heapq.nlargest(len(temptokens1), temptokens1)

        top_k_list[documents[doc_id].id] = []
        for (x, y) in tokindex:
            top_k_list[documents[doc_id].id].append((feature_names[y], x))
            # if isInChunk(feature_names[y],set(documents[doc_id].npchunks)) and not isAlreadyPresent(feature_names[y],top_k_list[documents[doc_id].id]) and len(feature_names[y]) > 2 :
            #     top_k_list[documents[doc_id].id].append((feature_names[y],x) )

        doc_id += 1

    for doc in documents:

        f = open(OUTPUT_DIR + OUTPUT_FOL + "/" + doc.id.replace("\\","_") + ".txt.phrases", 'w')
        writeformat = [ str(x)+","+str(round(y,4)) for (x,y) in top_k_list[doc.id]]
        f.write('\n'.join(writeformat[0:top_k]))
        f.write('\n')
        f.close()






def extract_vectors( documents,  ngram=(1,1), OUTPUT_FOL='UNIGRAMS'):
    '''
    Return the top K 1-gram terms according to TF-IDF
    Load corpus and convert to Dictionary and Corpus of gensim
    :param corpus_path
    :param num_feature, indicate how many terms you wanna retain, not useful now
    :return:
    '''

    if not os.path.exists(OUTPUT_DIR + OUTPUT_FOL):
        os.makedirs(OUTPUT_DIR + OUTPUT_FOL)

    corpus = [' '.join(preprocessText(document.text)) for document in documents]


    tf = TfidfVectorizer(analyzer='word', ngram_range=ngram,stop_words=stopwords,min_df=3)
    tfidf_matrix = tf.fit_transform(corpus)

    doc_id=0

    for doc in tfidf_matrix.todense():
        f = open(OUTPUT_DIR + OUTPUT_FOL + "/" + documents[doc_id].id.replace("\\","_") + ".txt.phrases", 'wb')
        pickle.dump(doc.tolist()[0],f)
        doc_id += 1
        f.close()


def extract_unigrams( documents,  ngram=3, OUTPUT_FOL='UNIGRAMS'):
    '''
    Return the top K 1-gram terms according to TF-IDF
    Load corpus and convert to Dictionary and Corpus of gensim
    :param corpus_path
    :param num_feature, indicate how many terms you wanna retain, not useful now
    :return:
    '''

    if not os.path.exists(OUTPUT_DIR + OUTPUT_FOL):
        os.makedirs(OUTPUT_DIR + OUTPUT_FOL)


    for doc in documents:
        l_ngrams = []
        tokens = preprocessText(doc.text)
        for i in range(0,ngram):
            l_ngrams += nltk.ngrams(tokens,n=i+1)

        tokens = set()
        for token in l_ngrams:
            tokens.add(' '.join(list(token)))

        f = open(OUTPUT_DIR + OUTPUT_FOL + "/" + doc.id.replace("\\","_") + ".txt.phrases", 'w')
        f.write('\n'.join(tokens))
        f.close()



def extract_author_keywords(keyword_trie, documents, OUTPUT_FOL):
    '''
    Return all the matching keywords according to the given keyword list (ACM/ACL/MS/Wiki/IR-glossary)
    :param keyword_trie:
    :param documents:
    :return:
    '''

    OUTPUT_FOLN = OUTPUT_DIR + OUTPUT_FOL
    if not os.path.exists(OUTPUT_FOLN):
        os.makedirs(OUTPUT_FOLN)

    print('Extracting keywords basing on wordlist, output to %s' % os.path.abspath(OUTPUT_FOLN))
    for doc in documents:

        f = open(OUTPUT_FOLN+'//'+doc.id+".txt.phrases",'w')
        keyword_list = keyword_trie.scan(doc.text)
        keyword_dict = Counter(keyword_list)
        f.write('\n'.join(keyword_dict.keys()))
        f.close()


def extract_lda(documents,testset,OUTPUT_FOL="wLDA",topics_n = 200):
    '''
    Extract the top K highest probability concepts according to the LLDA result
    :param documents:
    :param top_k:
    :return:
    '''
    # export_to_files(documents)
    doc_set=[]
    for doc in documents:
        doc_set.append(doc.text)

    tok_set=[]
    tok_dict={}
    test_dict = {}

    for doc in documents:
        tokens = preprocessText(doc.text,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS)
        tok_set.append(tokens)
        tok_dict[doc.id] = tokens

    for doc in testset:
        tokens = preprocessText(doc.text,stemming=IS_STEM,stopwords_removal=REMOVE_STOPWORDS)
        tok_set.append(tokens)
        test_dict[doc.id] = tokens

    dictionary = corpora.Dictionary(tok_set)
    corpus = [dictionary.doc2bow(text) for text in tok_set]

    OUTPUT_FOLN = OUTPUT_DIR + OUTPUT_FOL
    if not os.path.exists(OUTPUT_FOLN):
        os.makedirs(OUTPUT_FOLN)

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topics_n, id2word = dictionary)
    pickle.dump(ldamodel, open('data/ldagensim'+str(topics_n),'wb'))

    for doci in documents:
        doc = tok_dict[doci.id]
        f = open( OUTPUT_FOLN + "/" + doci.id + ".txt.phrases", 'wb')
        topics = sparse2full(ldamodel[dictionary.doc2bow(doc)], topics_n).tolist()

        pickle.dump(topics,f)
        f.close()

    for doci in testset:
        doc = test_dict[doci.id]
        f = open( OUTPUT_FOLN + "/" + doci.id + ".txt.phrases", 'wb')
        topics = sparse2full(ldamodel[dictionary.doc2bow(doc)], topics_n).tolist()

        pickle.dump(topics,f)
        f.close()

def extract_high_tfidf_words( documents, top_k=200, ngram=(1,1), OUTPUT_FOL='TFIDF'):
    '''
    Return the top K 1-gram terms according to TF-IDF
    Load corpus and convert to Dictionary and Corpus of gensim
    :param corpus_path
    :param num_feature, indicate how many terms you wanna retain, not useful now
    :return:
    '''

    if not os.path.exists(OUTPUT_DIR + OUTPUT_FOL):
        os.makedirs(OUTPUT_DIR + OUTPUT_FOL)

    texts = [' '.join(preprocessText(document.text,stemming=True,stopwords_removal=True))  for document in documents]

    #### Create Scikitlearn corpus
    top_k_list = {}


    tf = TfidfVectorizer(analyzer='word', ngram_range=ngram,stop_words=stopwords,min_df=2,max_df=1000)
    tfidf_matrix = tf.fit_transform(texts)
    feature_names = tf.get_feature_names()

    doc_id=0

    for doc in tfidf_matrix.todense():
        temptokens = zip(doc.tolist()[0], itertools.count())
        temptokens1=[]
        for (x, y) in temptokens:
            stemy = feature_names[y]
            if x > 0.0:
                temptokens1.append((x,y))

        tokindex = heapq.nlargest(len(temptokens1), temptokens1)

        top_k_list[documents[doc_id].id] = []
        for (x, y) in tokindex:
            top_k_list[documents[doc_id].id].append((feature_names[y],x) )
        doc_id += 1



    for doc in documents:

        # output_file.write('{0}\t{1}\n'.format(doc.id, ','.join(top_k_list[doc.id])))
        f = open(OUTPUT_DIR + OUTPUT_FOL + "/" + doc.id.replace("\\","_") + ".txt.phrases", 'w')
        writeformat = [ str(x)+","+str(round(y,4)) for (x,y) in top_k_list[doc.id]]
        f.write('\n'.join(writeformat[0:top_k]))
        # f.write('\n'.join(top_k_list[doc.id][0:top_k]))
        f.write('\n')
        f.close()



if __name__=='__main__':

    keyword_trie = None
    word_list = [ 'greedy-wiki']#,'gredy-acm']

    listbooks = ['irv-','issr-','mir-','iir-','foa-','zhai-','iirbookpubs-','seirip-','chapterwiseiir-','wiki-','wikitest-']
    # listbooks = [ 'iir-',  'wikitest-']
    listbooks_test = ['chapterwiseiir']

    documents = load_document(IR_CORPUS,listbooks)
    documentstest = load_document(IR_CORPUS,listbooks_test)

    # kl = KeywordList('greedy-acm')
    # keyword_trie = kl.trie
    # extract_author_keywords(keyword_trie, documents, 'greedy-acm')
    #


    kl = None
    kl = KeywordList('greedy-wiki')
    keyword_trie = kl.trie
    extract_author_keywords(keyword_trie, documents, 'greedy-wiki')



    ## Code For LDA
    # llistbooks = ['irv-','issr-','foa-','zhai-','seirip-','wiki-']
    # llistbooks_test = ['chapterwiseiir','wikitest-','mir-','iir-','iirbookpubs-']
    #
    # ldocuments = load_document(IR_CORPUS,llistbooks)
    # ldocumentstest = load_document(IR_CORPUS,llistbooks_test)
    # for i in range(200,300,50):
    #     extract_lda(ldocuments, OUTPUT_FOL="tLDA"+str(i),topics_n=i,testset=ldocumentstest)
    #
    # llistbooks = ['irv-', 'issr-', 'foa-', 'zhai-', 'seirip-', 'wiki-', 'sigir']
    # llistbooks_test = ['chapterwiseiir', 'wikitest-', 'mir-', 'iir-', 'iirbookpubs-']
    #
    # ldocuments = load_document(IR_CORPUS, llistbooks)
    # ldocumentstest = load_document(IR_CORPUS, llistbooks_test)
    # for i in range(200, 300, 50):
    #     extract_lda(ldocuments, OUTPUT_FOL="tsLDA" + str(i), topics_n=i, testset=ldocumentstest)

    # # Code For NP Chunks
    # print("NP Chunks ")
    # extract_np_high_tfidf_words( documents, top_k=10, ngram=(1,3), OUTPUT_FOL='TFIDFNP10')
    # extract_np_high_tfidf_words(documents, top_k=30, ngram=(1, 3), OUTPUT_FOL='TFIDFNP30')
    #
    #
    ## Code For extracting VSM
    # print("VSM")
    # extract_vectors(documents,  ngram=(1, 1), OUTPUT_FOL='UNIGRAM')
    #
    ## Code For Extract Unigrams
    # print("UNIGRAMS ")
    # extract_unigrams(documents,  ngram=(1, 1), OUTPUT_FOL='UNIGRAMS')
    #
    # ## Code For Extract Unigrams
    # print("NGRAMS ")
    # extract_unigrams(documents,  ngram=3, OUTPUT_FOL='NGRAMS')
    #

    # print("HIGH NP TFIDF WORDS")
    # for i in range(1,4):
    #     extract_np_high_tfidf_words( documents, top_k=5, ngram=(i,i), OUTPUT_FOL='TFIDFNP'+str(i))

    # print("HIGH TFIDF WORDS")
    # for i in range(1,4):
    #     extract_high_tfidf_words( documents, top_k=5, ngram=(i,i), OUTPUT_FOL='TFIDF'+str(i))

    # print("Global NGRAMS")
    # getGlobalngrams(grams=(2,2),documents=documents,threshold=0.01)
    # getGlobalngrams(grams=(3,3), documents=documents, threshold=0.01)

    # print("NGRAMS ")
    # listbooks = ['mir-','iir-','iirbookpubs-','chapterwiseiir-','wikitest-']
    # documents = load_document(IR_CORPUS,listbooks)
    #
    # extract_unigrams(documents,  ngram=3, OUTPUT_FOL='13NGRAMS')
