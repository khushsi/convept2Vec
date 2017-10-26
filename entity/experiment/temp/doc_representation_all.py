# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import pickle
import os,sys
import numpy as np
import sklearn
# from entity import experiment
import entity
# import matplotlib.pyplot as plt
from gensim.models import TfidfModel

from entity.config import Config
from entity.data import DataProvider

from gensim.models.ldamodel import LdaModel
from gensim import corpora
from experiment.gensim_data import load_review_text
from experiment.rank_metric import mean_average_precision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from tsne import bh_sne
# from tsne.bh_sne import BH_SNE
import brewer2mpl
import matplotlib as mpl
from scipy import spatial

mpl.use('TkAgg')

__author__ = "Khushboo Thaker"
__email__ = "kmt81@pitt.edu"

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
IS_STEMMING=False
IS_WEIGHTED=True

def myownstem(word):
    stemword= word

    if IS_STEMMING:
        stemword = stemmer.stem(word)

    return stemword

def multiwordstem(word):
    word_list = word.split(" ")
    for i in range(len(word_list)):
        word_list[i] = myownstem(word_list[i])
    return ' '.join(word_list)

tags_of_interest = [ "Data_pre-processing" , "Document_classification" , "Document_clustering" , "Evaluation_measures_(information_retrieval)"  , "Language_model" , "Probabilistic_relevance_model" , "Query_understanding" , "Relevance_feedback" , "Search_data_structure" , "Search_engine_indexing"  ,  "Tf–idf"]


model_name = 'glove' # lda, ntm, doc2vec, tfidf, attr
model_file='/Users/khushsi/Data/glove/glove.6B.200d.txt'


if __name__ == '__main__':
    print('Run experiment for %s' % model_name)
    ########################## Load the X-y pairs #############################
    print('Load the X-y pairs')
    print('is Stemmed',IS_STEMMING)
    print('is Weighted',IS_WEIGHTED)
    args = sys.argv
    # args = [args[0], "prodx_doc2vec", "prod", "200", "5"]
    args = [args[0], "", "prod", "200", "1"]
    # print(args)
    flag = args[1]
    n_processer = int(args[4])
    conf = Config(flag, args[2], int(args[3]))
    print(flag)


    home = os.environ["HOME"]
    home_path = home + "/Data/yelp/10-category-classification/"
    xy_path = home_path + 'yelp_class_Xid_Y.pkl'

    # glove_f = open(model_file,'r')
    doc_embed={}
    # for line in glove_f:
    #     wordlist = line.replace("\n","").split(" ")
    #     doc_embed[wordlist[0]]=np.array(wordlist[1:], dtype='float64')
    #     # print(doc_embed[wordlist[0].lower()])
    #     # print(wordlist[0])
    import pickle
    # pickle.dump(doc_embed,open('/Users/khushsi/Data/glove/glove.6B.200d.txt.pkl','wb'))
    doc_embed = pickle.load(open('/Users/khushsi/Data/glove/glove.6B.200d.txt.pkl','rb'))
    totallength = 0

    irfolders=["wUNIGRAM","UNIGRAM_NOSTEM","TFIDFNPWP1310","TFIDFNPWP1320","TFIDF1310","TFIDF1320","LDA10", "LDA20", "LDA30", "LDA40", "LDA50", "LDA60", "LDA70", "LDA80", "LDA90", "LDA100", "LDA110", "LDA120", "LDA130", "LDA140", "LDA150", "LDA160", "LDA170", "LDA180", "LDA190","wikiacm", "greedy-wiki", "LDA180", "wikimergeall", "wikiitalic", "wikiacm", "greedy-acm","CopyRNN","UNIGRAM_NOSTEM","UNIGRAM","TFIDFNPWP1310","CopyRNNRevised","TFIDFNPWP1305","TFIDF1210", "TFIDF1310"]
    combineresults={}
    for ir_directory in irfolders:
        print(ir_directory ,end=" ")
        ir_directory =  "/Users/khushsi/Downloads/concept_extraction/acm_dl/src/keyphrase_output_199/" + ir_directory +"/"
        print(ir_directory)
        fileobj = {}
        wikiobj = {}
        fcount={}
        wikifiles = [    "Controlled vocabulary.txt.phrases",
                        "Bias-variance tradeoff.txt.phrases",
                        "Bias-variance_tradeoff.phrases", "Controlled_vocabulary.txt.phrases", "Data compression.txt.phrases", "Data mining.txt.phrases", "Data modeling.txt.phrases", "Data pre-processing.txt.phrases", "Document classification.txt.phrases", "Document clustering.txt.phrases", "Evaluation measures (information retrieval).txt.phrases", "Evaluation of binary classifiers.txt.phrases", "Extended Boolean model.txt.phrases", "Feature Selection Toolbox.txt.phrases", "Feature selection.txt.phrases", "Heaps' law.txt.phrases", "Information retrieval.txt.phrases", "K-means clustering.txt.phrases", "K-nearest neighbors algorithm.txt.phrases", "Language model.txt.phrases", "Lemmatisation.txt.phrases", "Multiclass classification.txt.phrases", "Naive Bayes classifier.txt.phrases", "Nearest centroid classifier.txt.phrases", "Nearest neighbor.txt.phrases", "Phrase search.txt.phrases", "Probabilistic relevance model.txt.phrases", "Query understanding.txt.phrases", "Relevance feedback.txt.phrases", "Search data structure.txt.phrases", "Search engine indexing.txt.phrases", "Skip list.txt.phrases", "Speech disorder.txt.phrases", "Spell checker.txt.phrases", "Standard Boolean model.txt.phrases", "Stemming.txt.phrases", "Stop words.txt.phrases", "Text segmentation.txt.phrases", "Tf-idf.txt.phrases", "Vector space model.txt.phrases", "Wildcard character.txt.phrases", "Zipf's law.txt.phrases"]
        for filename in os.listdir(ir_directory):
            # if filename.endswith('phrases') and filename.startswith('iir'):
            if filename.endswith('phrases') and filename.startswith('iir') \
                    and not filename.startswith("iir_8_6") \
                    and not filename.startswith("iir_8_7") \
                    and not filename.startswith("iir 8 6") \
                    and not filename.startswith("iir 8 7") \
                    and not filename.startswith("iir_4") \
                    and not filename.startswith("iir 4") :

                filenamek = filename.split(".")[0].replace(" ","_")
                fileobj[filenamek] = np.array(np.zeros(200), dtype='float64')
                fcount[filenamek]=0
                with open(ir_directory+filename,'r') as irfile:
                    # print(irfile)
                    for line in irfile.readlines():
                        if(len(line.split(",")) >= 2):
                            wei =  float(line.split(",")[1].strip())
                        else:
                            wei = 1

                        line = line.split(",")[0]

                        concept = multiwordstem(' '.join(line.lower().split()))
                        concept = concept.replace(' ',"-")
                        # print (concept)
                        if(concept in doc_embed):
                            # print("found",concept)
                            if not IS_WEIGHTED:
                                wei = 1
                            fcount[filenamek] += 1
                            fileobj[filenamek] += (doc_embed[concept] * wei )
                        # else:
                            # print("not found",concept)
                # print(fileobj[filename])
                # print(fcount[filename])
                            # print(doc_embed[0][idx])

            if filename in wikifiles:
                filenamek = filename.split(".")[0].lower().replace("_"," ")
                wikiobj[filenamek] = np.array(np.zeros(200),dtype='float64')
                fcount[filenamek] = 0
                with open(ir_directory+filename,'r') as irfile:
                    # print(irfile)
                    for line in irfile.readlines():
                        if(len(line.split(",")) >= 2):
                            wei =  float(line.split(",")[1].strip())
                        else:
                            wei = 1
                        line = line.split(",")[0]

                        concept = multiwordstem(' '.join(line.lower().split()))
                        concept = concept.replace(' ', "-")
                        if(concept in doc_embed):
                            # print(" found  wiki",concept)
                            # print(doc_embed[concept])
                            # print(wikiobj[filenamek])
                            if not IS_WEIGHTED:
                                wei = 1
                            fcount[filenamek] += 1
                            wikiobj[filenamek] += (doc_embed[concept] * wei)
                        # else:
                            # print("not found",concept)
                # print(wikiobj[filename])
                # print(fcount[filename])
        # print(fcount)

        # print(doc_embed[0][:])
        docwikiscore = {}
        import queue as Q

        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)

        for doc in wikiobj.keys():
            if(fcount[doc] > 0) :
                # print(wikiobj[doc])
                wikiobj[doc] = np.divide(wikiobj[doc] , np.ones(200) * fcount[doc])
                # print(doc)
                # print(wikiobj[doc])
        #
        for doc in fileobj.keys():
            if(fcount[doc] > 0) :
                fileobj[doc] = np.divide(fileobj[doc] , np.ones(200) * fcount[doc])
                # print(doc)
                # print(fileobj[doc])
        # print(len(wikiobj.keys()),"wiki obj keys")
        for doc in fileobj.keys():
            docwikiscore[doc] = Q.PriorityQueue()
            for wiki in wikiobj.keys():
                docwikiscore[doc].put((spatial.distance.cosine(fileobj[doc],wikiobj[wiki] ) ,wiki))


        # print(prediction)

        wikiannotations = '../../wikiannotationssmall_multiple.csv'
        file = open(wikiannotations,'r')
        import csv
        reader = csv.reader(file)
        true = {}
        for row in reader:
            # print(row[0])
            results=row[2].replace("_"," ").lower().split(",")
            true[row[0]]=list(map(str.strip, results))

        # print(true)

        prediction = {}
        for doc in fileobj.keys():
            # icount = 0
            # Done=False
            # print(doc)
            prediction[doc] = []
            sizeofpred = docwikiscore[doc].qsize()
            for i in range(sizeofpred):
                prediction[doc].append(docwikiscore[doc].get(0)[1])

            # prediction[doc]=[docwikiscore[doc].get(0)[1],docwikiscore[doc].get(0)[1],docwikiscore[doc].get(0)[1],docwikiscore[doc].get(0)[1],docwikiscore[doc].get(0)[1],docwikiscore[doc].get(0)[1],docwikiscore[doc].get(0)[1],docwikiscore[doc].get(0)[1],docwikiscore[doc].get(0)[1],docwikiscore[doc].get(0)[1]]
            # print("--",prediction[doc])

        results=[]
        for keyd in prediction.keys():
            # print(keyd)
            # print(true[keyd])
            tempresult=[]
            if keyd in true.keys():
                pcount = 0
                for pred in prediction[keyd]:
                    # print(pred)
                    pcount = pcount + 1
                    if keyd not in combineresults.keys():
                        combineresults[keyd] = []
                    combineresults[keyd].append((pred, pcount))

                    if(pred in true[keyd]):
                        tempresult.append(1)
                    else:
                        tempresult.append(0)
                results.append(tempresult)

        # print(results)

        print(mean_average_precision(results))