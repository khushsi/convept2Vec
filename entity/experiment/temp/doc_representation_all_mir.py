# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import os,sys
import numpy as np
from entity.config import Config
from entity.rank_metric import mean_average_precision,ndcg_at_k
from scipy import spatial
import csv

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

    doc_embed = pickle.load(open('/Users/khushsi/Data/glove/glove.6B.200d.txt.pkl','rb'))
    totallength = 0

    irfolders = ["mUNIGRAM","mgTFIDF1340","mgTFIDFNPWP1340","mgTFIDFNPWP1330","mgTFIDFNPWP1310","mgTFIDFNPWP1320","mgTFIDF1330","mgTFIDF1310", "mgTFIDF1320"]

    combineresults={}
    for ir_directory in irfolders:
        print(ir_directory ,end=" ")
        ir_directory =  "/Users/khushsi/Downloads/concept_extraction/acm_dl/src/keyphrase_output/" + ir_directory +"/"
        print(ir_directory)
        fileobj = {}
        wikiobj = {}
        fcount={}

        wikifiles = ["mir-0117.txt.phrases", "mir-0118.txt.phrases", "mir-0119.txt.phrases", "mir-0118.txt.phrases",
                     "mir-0119.txt.phrases", "mir-0120.txt.phrases", "mir-0121.txt.phrases", "mir-0117.txt.phrases",
                     "mir-0119.txt.phrases", "mir-0120.txt.phrases", "mir-0119.txt.phrases", "mir-0121.txt.phrases",
                     "mir-0121.txt.phrases", "mir-0150.txt.phrases", "mir-0136.txt.phrases", "mir-0196.txt.phrases",
                     "mir-0133.txt.phrases", "mir-0160.txt.phrases", "mir-0056.txt.phrases", "mir-0057.txt.phrases",
                     "mir-0061.txt.phrases", "mir-0062.txt.phrases", "mir-0063.txt.phrases", "mir-0064.txt.phrases",
                     "mir-0059.txt.phrases", "mir-0060.txt.phrases", "mir-0058.txt.phrases", "mir-0059.txt.phrases",
                     "mir-0060.txt.phrases", "mir-0059.txt.phrases", "mir-0059.txt.phrases", "mir-0060.txt.phrases",
                     "mir-0057.txt.phrases", "mir-0060.txt.phrases", "mir-0057.txt.phrases", "mir-0066.txt.phrases",
                     "mir-0057.txt.phrases", "mir-0199.txt.phrases", "mir-0084.txt.phrases", "mir-0083.txt.phrases",
                     "mir-0092.txt.phrases", "mir-0089.txt.phrases", "mir-0095.txt.phrases", "mir-0096.txt.phrases",
                     "mir-0082.txt.phrases", "mir-0084.txt.phrases", "mir-0089.txt.phrases", "mir-0092.txt.phrases",
                     "mir-0085.txt.phrases", "mir-0086.txt.phrases", "mir-0087.txt.phrases", "mir-0083.txt.phrases",
                     "mir-0084.txt.phrases", "mir-0088.txt.phrases", "mir-0084.txt.phrases", "mir-0089.txt.phrases",
                     "mir-0092.txt.phrases", "mir-0093.txt.phrases", "mir-0094.txt.phrases", "mir-0093.txt.phrases",
                     "mir-0094.txt.phrases", "mir-0085.txt.phrases", "mir-0090.txt.phrases", "mir-0091.txt.phrases",
                     "mir-0093.txt.phrases", "mir-0094.txt.phrases", "mir-0093.txt.phrases", "mir-0094.txt.phrases",
                     "mir-0092.txt.phrases", "mir-0096.txt.phrases", "mir-0095.txt.phrases", "mir-0030.txt.phrases",
                     "mir-0030.txt.phrases", "mir-0030.txt.phrases", "mir-0030.txt.phrases", "mir-0030.txt.phrases",
                     "mir-0030.txt.phrases", "mir-0030.txt.phrases", "mir-0030.txt.phrases", "mir-0055.txt.phrases",
                     "mir-0135.txt.phrases", "mir-0117.txt.phrases", "mir-0116.txt.phrases", "mir-0119.txt.phrases",
                     "mir-0196.txt.phrases", "mir-0135.txt.phrases", "mir-0030.txt.phrases", "mir-0084.txt.phrases"]

        for filename in os.listdir(ir_directory):
            # if filename.endswith('phrases') and filename.startswith('iir'):
            if filename.endswith('phrases') and filename.startswith('iir') :

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

            # if filename in wikifiles:
            if filename.endswith('phrases') and filename.startswith('mir'):

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

        print("Calculated Cosine Similarity")
        wikiannotations = '../../textbook_textbook.txt'
        file = open(wikiannotations,'r')
        reader = csv.reader(file,delimiter="\t")
        true = {}
        truescore={}

        for row in reader:
            results=row[1].split(",")
            true[row[0]] = []
            truescore[row[0]] ={}
            for val in results:
                text=val.split(":")[0].replace(".txt.phrases","")
                score = float(val.split(":")[1])
                true[row[0]].append(text)
                truescore[row[0]].update({text:score})

        prediction = {}
        for doc in fileobj.keys():
            prediction[doc] = []
            sizeofpred = docwikiscore[doc].qsize()
            for i in range(sizeofpred):
                prediction[doc].append(docwikiscore[doc].get(0)[1])

        results = []
        ndcg = 0
        ndcg1 = 0
        ncount = 0
        for keyd in prediction.keys():
            tempresult = []
            if keyd in true.keys():
                pcount = 0
                ncount += 1
                for pred in prediction[keyd]:
                    pcount = pcount + 1
                    if keyd not in combineresults.keys():
                        combineresults[keyd] = []
                    combineresults[keyd].append((pred, pcount))

                    if (pred in true[keyd]):
                        obj = truescore[keyd]
                        score = obj[pred]
                        tempresult.append(score)
                    else:
                        tempresult.append(0)
                ndcg += ndcg_at_k(tempresult, 3, method=1)
                ndcg1 += ndcg_at_k(tempresult, 1, method=1)

                results.append(tempresult)

        if (ncount > 0):
            # print(ndcg,ncount)

            print(ndcg / ncount)
            print(ndcg1 / ncount)
            # print(results)