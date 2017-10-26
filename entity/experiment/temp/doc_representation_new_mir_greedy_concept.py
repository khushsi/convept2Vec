# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import os,sys
import numpy as np
from entity.config import Config
from entity.data import DataProvider
from scipy import spatial
from entity.rank_metric import mean_average_precision,ndcg_at_k
import pickle
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


model_name = 'doc2vec' # lda, ntm, doc2vec, tfidf, attr
model_dirs = {'lda':'ntm_model.freq=100.word=22548.lr=0.01', 'ntm':'ntm_model.freq=100.word=22548.lr=0.01', 'doc2vec':'prodx_doc2vec', 'tfidf':'ntm_model.freq=100.word=22548.lr=0.01', 'attr':'attribute-argumented_model.freq=10'}



if __name__ == '__main__':
    print("hi")
    print('Run experiment for %s' % model_name)
    ########################## Load the X-y pairs #############################
    print('Load the X-y pairs')
    print('is Stemmed',IS_STEMMING)
    print('is Weighted',IS_WEIGHTED)
    args = sys.argv
    # args = [args[0], "prodx_doc2vec", "prod", "200", "5"]
    args = [args[0], model_dirs[model_name], "prod", "200", "1"]
    # print(args)
    flag = args[1]
    n_processer = int(args[4])
    conf = Config(flag, args[2], int(args[3]))
    print(flag)


    home = os.environ["HOME"]
    dp = DataProvider(conf)
    doc_embed = np.load(conf.path_doc_npy+'.npy')
    totallength = 0

    wordembeddingfile = 'wordEmp.txt'
    gwordembeddingfile = 'glove_model2.txt'
    f = open(wordembeddingfile,'w')
    for i in range(len(dp.idx2prod)):
        f.write(dp.idx2prod[i].replace(" ","_")+" "+' '.join(map(str, doc_embed[0][i]))+"\n")

    from gensim.models import Word2Vec,keyedvectors,KeyedVectors

    model = KeyedVectors.load_word2vec_format(gwordembeddingfile, binary=False)



    wikifiles = ["mir-0117.txt.phrases", "mir-0118.txt.phrases", "mir-0119.txt.phrases", "mir-0118.txt.phrases", "mir-0119.txt.phrases", "mir-0120.txt.phrases", "mir-0121.txt.phrases", "mir-0117.txt.phrases", "mir-0119.txt.phrases", "mir-0120.txt.phrases", "mir-0119.txt.phrases", "mir-0121.txt.phrases", "mir-0121.txt.phrases", "mir-0150.txt.phrases", "mir-0136.txt.phrases", "mir-0196.txt.phrases", "mir-0133.txt.phrases", "mir-0160.txt.phrases", "mir-0056.txt.phrases", "mir-0057.txt.phrases", "mir-0061.txt.phrases", "mir-0062.txt.phrases", "mir-0063.txt.phrases", "mir-0064.txt.phrases", "mir-0059.txt.phrases", "mir-0060.txt.phrases", "mir-0058.txt.phrases", "mir-0059.txt.phrases", "mir-0060.txt.phrases", "mir-0059.txt.phrases", "mir-0059.txt.phrases", "mir-0060.txt.phrases", "mir-0057.txt.phrases", "mir-0060.txt.phrases", "mir-0057.txt.phrases", "mir-0066.txt.phrases", "mir-0057.txt.phrases", "mir-0199.txt.phrases", "mir-0084.txt.phrases", "mir-0083.txt.phrases", "mir-0092.txt.phrases", "mir-0089.txt.phrases", "mir-0095.txt.phrases", "mir-0096.txt.phrases", "mir-0082.txt.phrases", "mir-0084.txt.phrases", "mir-0089.txt.phrases", "mir-0092.txt.phrases", "mir-0085.txt.phrases", "mir-0086.txt.phrases", "mir-0087.txt.phrases", "mir-0083.txt.phrases", "mir-0084.txt.phrases", "mir-0088.txt.phrases", "mir-0084.txt.phrases", "mir-0089.txt.phrases", "mir-0092.txt.phrases", "mir-0093.txt.phrases", "mir-0094.txt.phrases", "mir-0093.txt.phrases", "mir-0094.txt.phrases", "mir-0085.txt.phrases", "mir-0090.txt.phrases", "mir-0091.txt.phrases", "mir-0093.txt.phrases", "mir-0094.txt.phrases", "mir-0093.txt.phrases", "mir-0094.txt.phrases", "mir-0092.txt.phrases", "mir-0096.txt.phrases", "mir-0095.txt.phrases", "mir-0030.txt.phrases", "mir-0030.txt.phrases", "mir-0030.txt.phrases", "mir-0030.txt.phrases", "mir-0030.txt.phrases", "mir-0030.txt.phrases", "mir-0030.txt.phrases", "mir-0030.txt.phrases", "mir-0055.txt.phrases", "mir-0135.txt.phrases", "mir-0117.txt.phrases", "mir-0116.txt.phrases", "mir-0119.txt.phrases", "mir-0196.txt.phrases", "mir-0135.txt.phrases", "mir-0030.txt.phrases", "mir-0084.txt.phrases"]
   # irfolders=["TFIDF_MIR","gold_E2v", "greedy-acm", "greedy-h", "LDA10", "LDA20", "LDA30", "LDA40", "LDA50", "LDA60", "LDA70", "LDA80", "LDA90", "LDA100", "LDA110", "LDA120", "LDA130", "LDA140", "LDA150", "LDA160", "LDA170", "LDA180", "LDA190", "TFIDF23", "TFIDF1105", "TFIDF1110", "TFIDF1205", "TFIDF1210", "TFIDF1305", "TFIDF1310", "TFIDF2205", "TFIDF2210", "TFIDF2310", "TFIDF3305", "TFIDF3310", "TFIDF1320","TFIDF1220","TFIDFNPWP3", "TFIDFNPWP23", "TFIDFNPWP1305", "TFIDFNPWP1310", "TFIDFNPWP2205", "TFIDFNPWP2210", "TFIDFNPWP2310", "TFIDFNPWP3305", "TFIDFNPWP3310","TFIDFNPWP1320", "wikiacm", "wikiitalic", "wikilink", "wikimergeall", "wikisection"]
   # irfolders=[ "gTFIDF1310","gTFIDF1320","wikimergeall","wikiacm","greedy-acm", "UNIGRAM_NOSTEM","TFIDFNPWP1310","TFIDFNPWP1320","TFIDF1310","TFIDF1320","wikiacm","LDA10", "LDA20", "LDA30", "LDA40", "LDA50", "LDA60", "LDA70","UNIGRAM_NOSTEM","UNIGRAM","TFIDFNPWP1310","gTFIDFNPWP1310","CopyRNNRevised","TFIDFNPWP1305","gTFIDFNPWP1305", "greedy-wiki", "TFIDF1210", "TFIDF1310","gTFIDF1310","LDA180", "gold_E2v", "wikimergeall", "wikiitalic", "wikiacm", "greedy-acm","CopyRNN"]
    irfolders = ["mgTFIDF1340",'gTFIDF1310']#,"mgTFIDFNPWP1340","mgTFIDFNPWP1330","mgTFIDFNPWP1310","mgTFIDFNPWP1320","mgTFIDF1330","mgTFIDF1310", "mgTFIDF1320"]
    irfolders = ["iLDA100","iLDA50","iLDA150","iLDA200","iLDA250","CopyRNN","TFIDFNPWP1310", "greedy-wiki","TFIDF1210","CopyRNN","gold_E2v","LDA180"]
    combineresults={}
    for ir_directory in irfolders:
        print(ir_directory ,end=" ")
        # ir_directory =  "/Users/khushsi/Downloads/concept_extraction/acm_dl/src/keyphrase_output_before20Aug/" + ir_directory +"/"
        ir_directory = "/Users/khushsi/Downloads/concept_extraction/acm_dl/src/keyphrase_output_199/" + ir_directory + "/"
        print(ir_directory)
        fileobj = {}
        wikiobj = {}
        fileobjtext = {}
        wikiobjtext = {}
        fcount={}


        # print("I am here")

        for filename in os.listdir(ir_directory):
            # if filename.endswith('phrases') and filename.startswith('iir'):
            if filename.endswith('phrases') and filename.startswith('iir')  :

                filenamek = filename.split(".")[0].replace(" ","_")
                fileobj[filenamek] = np.zeros(200)
                fileobjtext[filenamek] = ""
                fcount[filenamek]=0
                with open(ir_directory+filename,'r') as irfile:
                    # print(irfile)
                    for line in irfile.readlines():
                        if(len(line.split(",")) >= 2):
                            wei =  float(line.split(",")[1].strip())
                        else:
                            wei = 1
                        line = line.split(",")[0]

                        # print(line)
                        concept = multiwordstem(' '.join(line.lower().split()))
                        # print (concept)
                        if(concept in dp.idx2prod):
                            # print("found",concept)
                            fileobjtext[filenamek] += " "+concept.replace(" ","_")

                            idx = dp.idx2prod.tolist().index(concept)
                            fcount[filenamek] += 1
                            if not IS_WEIGHTED:
                                wei = 1
                            fileobj[filenamek] += (doc_embed[0][idx] * wei)
                            # else:
                            # print("not found",concept)
                # print(fileobj[filename])
                # print(fcount[filename])
                            # print(doc_embed[0][idx])

            # if filename.endswith('phrases') and filename.startswith("mir") :
            if filename.endswith('phrases') and filename.startswith("mir"):
                # print(filename)
                filenamek = filename.split(".")[0].replace(" ","_")
                wikiobj[filenamek] = np.zeros(200)
                wikiobjtext[filenamek] = ""
                fcount[filenamek] = 0
                with open(ir_directory+filename,'r') as irfile:
                    # print(irfile)
                    for line in irfile.readlines():
                        if(len(line.split(",")) >= 2):
                            wei =  float(line.split(",")[1].strip())
                        else:
                            wei = 1
                        line = line.split(",")[0]

                        # print(line)
                        concept = multiwordstem(' '.join(line.lower().split()))

                        if(concept in dp.idx2prod):
                            wikiobjtext[filenamek] += " " + concept.replace(" ", "_")
                            # print("wiki found ",concept)
                            fcount[filenamek] += 1
                            idx = dp.idx2prod.tolist().index(concept)
                            if not IS_WEIGHTED:
                                wei = 1
                            wikiobj[filenamek] += ( doc_embed[0][idx] * wei)
                    # else:
                            # print("not found",concept)
                # print(wikiobj[filename])
                # print(fcount[filename])
        # print(fcount)

        # print(doc_embed[0][:])
        docwikiscore = {}
        docwikiscore2 = {}
        import queue as Q

        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)

        for doc in wikiobj.keys():
            if(fcount[doc] > 0) :
                wikiobj[doc] = np.divide(wikiobj[doc] , np.ones(200) * fcount[doc])
                        #
        for doc in fileobj.keys():
            if(fcount[doc] > 0) :
                fileobj[doc] = np.divide(fileobj[doc] , np.ones(200) * fcount[doc])

        # print(len(wikiobj.keys()),"wiki obj keys")
        VSMDIR = '/Users/khushsi/Downloads/concept_extraction/acm_dl/src/keyphrase_output/gVSM/'
        for filename in fileobj:
            fileobj[filename] = fileobj[filename].tolist() + pickle.load(open(VSMDIR+filename+".txt.phrases",'rb'))

        for filename in wikiobj:
            wikiobj[filename] = wikiobj[filename].tolist() + pickle.load(open(VSMDIR+filename+".txt.phrases",'rb'))

        for doc in fileobj.keys():
            docwikiscore2[doc] = Q.PriorityQueue()
            docwikiscore[doc] = Q.PriorityQueue()
            for wiki in wikiobj.keys():
                docwikiscore[doc].put((spatial.distance.cosine(fileobj[doc],wikiobj[wiki] ) ,wiki))
                docwikiscore2[doc].put((model.wmdistance(fileobjtext[doc].split(),wikiobjtext[wiki].split() ) ,wiki))


        wikiannotations = '../../textbook_textbook.txt'
        file = open(wikiannotations,'r')
        import csv
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

        # print(true)

        prediction = {}
        for doc in fileobj.keys():
            prediction[doc] = []
            sizeofpred = docwikiscore[doc].qsize()
            for i in range(sizeofpred):
                prediction[doc].append(docwikiscore[doc].get(0)[1])
            # if(doc in true.keys()):
            #     print(doc)
            #     print(prediction[doc])

        prediction2 = {}
        for doc in fileobj.keys():
            prediction2[doc] = []
            sizeofpred = docwikiscore2[doc].qsize()
            for i in range(sizeofpred):
                prediction2[doc].append(docwikiscore2[doc].get(0)[1])
            print(prediction2[doc])
            # if(doc in true.keys()):
        pickle.dump(prediction2,open("prediction2.pkl","wb"))

        results=[]
        ndcg = 0
        ndcg1 = 0
        ncount = 0
        for keyd in prediction.keys():
            tempresult=[]
            if keyd in true.keys():
                pcount = 0
                ncount += 1
                for pred in prediction[keyd]:
                    pcount = pcount + 1
                    if keyd not in combineresults.keys():
                        combineresults[keyd] = []
                    combineresults[keyd].append((pred, pcount))

                    if(pred in true[keyd]):
                        obj = truescore[keyd]
                        score = obj[pred]
                        tempresult.append(score)
                    else:
                        tempresult.append(0)

                ndcg += ndcg_at_k(tempresult,3,method=1)
                ndcg1 += ndcg_at_k(tempresult, 1, method=1)
                results.append(tempresult)

        if(ncount > 0):
            # print(ndcg,ncount)
            print(ndcg/ncount)
            print(ndcg1/ncount)

        results = []
        ndcg = 0
        ndcg1 = 0
        ncount = 0
        for keyd in prediction2.keys():
            tempresult = []
            if keyd in true.keys():
                pcount = 0
                ncount += 1
                for pred in prediction2[keyd]:
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
            print("wm",ndcg / ncount)
            print("wm",ndcg1 / ncount)

                # print(results)
        # print(ndcg_at_k(results, 1, method=1))
        # print(ndcg_at_k(results, 3, method=1))
        # print(ndcg_at_k(results, 5, method=1))
        # print(mean_average_precision(results))