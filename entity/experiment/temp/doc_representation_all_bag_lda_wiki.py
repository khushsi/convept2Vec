# -*- coding: utf-8 -*-
import os,sys
import numpy as np
from scipy import spatial
from entity.rank_metric import ndcg_at_k, mean_average_precision
import csv

__author__ = "Khushboo Thaker"
__email__ = "kmt81@pitt.edu"


# tags_of_interest = [ "Data_pre-processing" , "Document_classification" , "Document_clustering" , "Evaluation_measures_(information_retrieval)"  , "Language_model" , "Probabilistic_relevance_model" , "Query_understanding" , "Relevance_feedback" , "Search_data_structure" , "Search_engine_indexing"  ,  "Tf–idf"]


if __name__ == '__main__':
    print('Run experiment for %s' % "vector space model")
    ########################## Load the X-y pairs #############################
    print('Load the X-y pairs')


    import pickle

    irfolders=["iLDA100","iLDA150","iLDA200","iLDA250","iLDA10","iLDA60","iLDA110","iLDA160","iLDA210","iLDA260","gVSM","gCVSM","iLDA10"]
    combineresults={}
    for ir_directory in irfolders:
        print(ir_directory ,end=" ")
        ir_directory =  "/Users/khushsi/Downloads/concept_extraction/acm_dl/src/keyphrase_output_199/" + ir_directory +"/"
        print(ir_directory)
        fileobj = {}
        wikiobj = {}
        fcount={}


        for filename in os.listdir(ir_directory):
            # if filename.endswith('phrases') and filename.startswith('iir'):
            if filename.endswith('phrases') and filename.startswith('iir') :
                filenamek = filename.split(".")[0].lower()
                fileobj[filenamek] = pickle.load(open(ir_directory+filename,'rb'))


            if filename.endswith('phrases') and not filename.startswith('iir') :
                filenamek = filename.split(".")[0].lower()
                wikiobj[filenamek] = pickle.load(open(ir_directory+filename,'rb'))


        docwikiscore = {}
        import queue as Q

        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)

        for doc in fileobj.keys():
            docwikiscore[doc] = Q.PriorityQueue()
            for wiki in wikiobj.keys():
                docwikiscore[doc].put((spatial.distance.cosine(fileobj[doc],wikiobj[wiki] ) ,wiki))


        print("Calculated Cosine Similarity")
        # wikiannotations = '../../textbook_textbook.txt'
        wikiannotations = '../../wikiannotationssmall_multiple.csv'
        file = open(wikiannotations,'r')
        reader = csv.reader(file,delimiter=",")
        true = {}

        for row in reader:

            results=row[2].replace("_"," ").lower().split(",")
            true[row[0]]=list(map(str.strip, results))

        prediction = {}
        for doc in fileobj.keys():
            prediction[doc] = []
            sizeofpred = docwikiscore[doc].qsize()
            for i in range(sizeofpred):
                prediction[doc].append(docwikiscore[doc].get(0)[1])

        results=[]
        for keyd in prediction.keys():
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
