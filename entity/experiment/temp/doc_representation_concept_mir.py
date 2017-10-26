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
from experiment.rank_metric import mean_average_precision, ndcg_at_k
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

def prepare_10class_dataset(dp, xy_path):
    '''
    Extract the data of interest
    :param dp:
    :param xy_path:
    :return:
    '''
    if os.path.exists(xy_path):
        with open(xy_path, 'rb') as f:
            X_idx, business_idx, Y, Y_name, Y_original = pickle.load(f)
    else:


        X_idx = []
        Y = []
        Y_name = []
        Y_original = []

        for doc_id, doc_tag_list in enumerate(dp.doc_tag_cor_fmatrix):
            # only keep the businesses that have only one label of interest
            class_id = -1
            count_found = 0
            for id, (t,t_id) in enumerate(tag_dict_of_interest.items()):
                # t_id is the original tag id, we map it into 0-9
                if doc_tag_list[t_id]:
                    count_found += 1
                    class_id = id
                    class_name = t
                    original_class_id = t_id
            if count_found == 1:
                X_idx.append(doc_id)
                Y.append(class_id)
                Y_name.append(class_name)
                Y_original.append(original_class_id)
        Y = np.asarray(Y)
        Y_name = np.asarray(Y_name)
        Y_original = np.asarray(Y_original)

        business_id_dict = {}
        for id_, business_id in enumerate(dp.idx2prod):
            business_id_dict[id_] = business_id
        business_idx = [business_id_dict[x_] for x_ in X_idx]

        with open(xy_path, 'wb') as f:
            pickle.dump([X_idx, business_idx, Y, Y_name, Y_original], f)

    return X_idx, business_idx, Y, Y_name, Y_original


def classify(X_train, Y_train, X_test, Y_test):
    from sklearn import svm

    clf = svm.LinearSVC(C = 1.0, class_weight = 'balanced', dual = True, fit_intercept = True,
    intercept_scaling = 1, loss = 'squared_hinge', max_iter = 1000,
    multi_class = 'ovr', penalty = 'l2', random_state = None, tol = 0.0001,
    verbose = 0)

    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    f_micro = sklearn.metrics.f1_score(Y_test, Y_pred, average='micro')
    p_micro = sklearn.metrics.precision_score(Y_test, Y_pred, average='micro')
    r_micro = sklearn.metrics.recall_score(Y_test, Y_pred, average='micro')

    f_macro = sklearn.metrics.f1_score(Y_test, Y_pred, average='macro')
    p_macro = sklearn.metrics.precision_score(Y_test, Y_pred, average='macro')
    r_macro = sklearn.metrics.recall_score(Y_test, Y_pred, average='macro')

    for i in range(len(X_test)):
        print (Y_test[i]),print(Y_pred[i])

    accuracy = sklearn.metrics.accuracy_score(Y_test, Y_pred)

    print('Accuracy=%f' % accuracy)

    print('*' * 10 + ' Micro Score ' + '*' * 10)
    print('p=%f' % p_micro)
    print('r=%f' % r_micro)
    print('f-score=%f' % f_micro)

    print('*' * 10 + ' Macro Score ' + '*' * 10)
    print('p=%f' % p_macro)
    print('r=%f' % r_macro)
    print('f-score=%f' % f_macro)

def classify2(X, y):
    from sklearn import svm

    clf = svm.LinearSVC(C = 1.0, dual = True, fit_intercept = True,
    intercept_scaling = 1, loss = 'squared_hinge', max_iter = 1000,
    multi_class = 'ovr', penalty = 'l2', random_state = None, tol = 0.0001,
    verbose = 0)
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    pred=[]
    target=[]
    for train_index, test_index in loo.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        Y_pred = clf.predict(X_test)

        pred.append(Y_pred)
        target.append(y_test)




    # print (target)

    f_micro = sklearn.metrics.f1_score(target, pred, average='micro')
    p_micro = sklearn.metrics.precision_score(target, pred, average='micro')
    r_micro = sklearn.metrics.recall_score(target, pred, average='micro')

    # f_macro = sklearn.metrics.f1_score(target, pred, average='macro')
    # p_macro = sklearn.metrics.precision_score(target, pred, average='macro')
    # r_macro = sklearn.metrics.recall_score(target, pred, average='macro')


    accuracy = sklearn.metrics.accuracy_score(target, pred)

    print('Accuracy=%f' % accuracy)

    print('*' * 10 + ' Micro Score ' + '*' * 10)
    print('p=%f' % p_micro)
    print('r=%f' % r_micro)
    print('f-score=%f' % f_micro)

    # print('*' * 10 + ' Macro Score ' + '*' * 10)
    # print('p=%f' % p_macro)
    # print('r=%f' % r_macro)
    # print('f-score=%f' % f_macro)

def visualize(x_data, y_data, y_name):

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = np.asarray(x_data).astype('float64')
    y_data = np.asarray(y_data).astype('int')
    y_name = np.asarray(y_name)
    x_data = x_data.reshape((x_data.shape[0], -1))

    # perform t-SNE embedding
    vis_data = bh_sne(x_data,perplexity=5)

    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]
    plt.interactive(False)

    fig, ax = plt.subplots()

    almost_black = '#262626'
    # set2 = brewer2mpl.get_map('Set3', 'qualitative', 10).mpl_colors
    set2 = plt.cm.Set3(np.linspace(0, 1, 10))

    for class_i in range(10):
        idx = np.where(y_data == class_i)[0]
        # print(idx)
        color = set2[class_i]
        # print('label=%s' % y_name[y])
        plt.scatter(vis_x[idx], vis_y[idx], label=y_name[class_i], alpha=0.9, edgecolor=almost_black, linewidth=0.15, facecolor=color)#s=0.5, cmap=plt.cm.get_cmap("jet", 10))
    # plt.colorbar(ticks=range(10))
    ax.legend(loc=1)
    ax.grid(True)

    plt.clim(-0.5, 9.5)
    plt.show()

model_name = 'doc2vec' # lda, ntm, doc2vec, tfidf, attr
model_dirs = {'lda':'ntm_model.freq=100.word=22548.lr=0.01', 'ntm':'ntm_model.freq=100.word=22548.lr=0.01', 'doc2vec':'prodx_doc2vec', 'tfidf':'ntm_model.freq=100.word=22548.lr=0.01', 'attr':'attribute-argumented_model.freq=10'}


def prepare_visualization():
    data_sample_dict = {}
    for x,y in zip(X,Y):
        if len(data_sample_dict.get(y, [])) < 1000:
            d = data_sample_dict.get(y, [])
            d.append(x)
            data_sample_dict[y] = d

    Y_name_dict = {}
    for i, n in enumerate(tags_of_interest):
        Y_name_dict[i] = n

    X_sample = []
    Y_sample = []
    Y_name_sample = []
    for y, x_list in data_sample_dict.items():
        for x in x_list:
            X_sample.append(x)
            Y_sample.append(y)
            Y_name_sample.append(Y_name_dict[y])
    return X_sample, Y_sample

if __name__ == '__main__':
    print('Run experiment for %s' % model_name)
    ########################## Load the X-y pairs #############################
    print('Load the X-y pairs')
    args = sys.argv
    # args = [args[0], "prodx_doc2vec", "prod", "200", "5"]
    args = [args[0], model_dirs[model_name], "prod", "200", "1"]
    # print(args)
    flag = args[1]
    n_processer = int(args[4])
    conf = Config(flag, args[2], int(args[3]))
    print(flag)


    home = os.environ["HOME"]
    home_path = home + "/Data/yelp/10-category-classification/"
    xy_path = home_path + 'yelp_class_Xid_Y.pkl'

    dp = DataProvider(conf)
    doc_embed = np.load(conf.path_doc_npy+'.npy')
    # word_embed = np.load(conf.path_word_npy+'.npy')
    # model_weights = np.load(conf.path_model_npy+'.npy')
    # print(dp.idx2word[1:100])
    # print(dp.idx2prod[573])
    print(len(dp.idx2prod))
    totallength = 0

    irfolders=["gold_E2v", "greedy-acm", "greedy-wiki", "LDA10", "LDA20", "LDA30", "LDA40", "LDA50", "LDA60", "LDA70", "LDA80", "LDA90", "LDA100", "LDA110", "LDA120", "LDA130", "LDA140", "LDA150", "LDA160", "LDA170", "LDA180", "LDA190", "TFIDF23", "TFIDF1105", "TFIDF1110", "TFIDF1205", "TFIDF1210", "TFIDF1305", "TFIDF1310", "TFIDF2205", "TFIDF2210", "TFIDF2310", "TFIDF3305", "TFIDF3310", "TFIDF1320","TFIDF1220","TFIDFNPWP3", "TFIDFNPWP23", "TFIDFNPWP1305", "TFIDFNPWP1310", "TFIDFNPWP2205", "TFIDFNPWP2210", "TFIDFNPWP2310", "TFIDFNPWP3305", "TFIDFNPWP3310","TFIDFNPWP1320", "wikiacm", "wikiitalic", "wikilink", "wikimergeall", "wikisection"]
    irfolders=["UNIGRAM","TFIDFNPWP1310","CopyRNNRevised","TFIDFNPWP1305", "greedy-wiki", "TFIDF1210", "TFIDF1310", "LDA180", "gold_E2v", "wikimergeall", "wikiitalic", "wikiacm", "greedy-acm","CopyRNN"]

    # irfolders=["TFIDFNPWP1305", "greedy-wiki", "TFIDF1210", "LDA180", "gold_E2v", "wikimergeall", "wikiitalic", "wikiacm", "greedy-acm"]
    # irfolders = ["TFIDFNPWP1305", "greedy-wiki","TFIDF1210"]
    combineresults={}
    for ir_directory in irfolders:
        print(ir_directory ,end=" ")
        ir_directory =  "/Users/khushsi/Downloads/concept_extraction/acm_dl/src/keyphrase_output_before20Aug/" + ir_directory +"/"
        fileobj = {}
        wikiobj = {}
        fcount={}
        wikifiles = ["Bias-variance tradeoff.txt.phrases","Bias-variance_tradeoff.phrases", "Controlled_vocabulary.txt.phrases", "Data compression.txt.phrases", "Data mining.txt.phrases", "Data modeling.txt.phrases", "Data pre-processing.txt.phrases", "Document classification.txt.phrases", "Document clustering.txt.phrases", "Evaluation measures (information retrieval).txt.phrases", "Evaluation of binary classifiers.txt.phrases", "Extended Boolean model.txt.phrases", "Feature Selection Toolbox.txt.phrases", "Feature selection.txt.phrases", "Heaps' law.txt.phrases", "Information retrieval.txt.phrases", "K-means clustering.txt.phrases", "K-nearest neighbors algorithm.txt.phrases", "Language model.txt.phrases", "Lemmatisation.txt.phrases", "Multiclass classification.txt.phrases", "Naive Bayes classifier.txt.phrases", "Nearest centroid classifier.txt.phrases", "Nearest neighbor.txt.phrases", "Phrase search.txt.phrases", "Probabilistic relevance model.txt.phrases", "Query understanding.txt.phrases", "Relevance feedback.txt.phrases", "Search data structure.txt.phrases", "Search engine indexing.txt.phrases", "Skip list.txt.phrases", "Speech disorder.txt.phrases", "Spell checker.txt.phrases", "Standard Boolean model.txt.phrases", "Stemming.txt.phrases", "Stop words.txt.phrases", "Text segmentation.txt.phrases", "Tf-idf.txt.phrases", "Vector space model.txt.phrases", "Wildcard character.txt.phrases", "Zipf's law.txt.phrases","Bias-variance tradeoff.phrases","Bias-variance_tradeoff.phrases", "Controlled_vocabulary.txt.phrases", "Data compression.txt.phrases", "Data mining.txt.phrases", "Data modeling.txt.phrases", "Data pre-processing.txt.phrases", "Document classification.txt.phrases", "Document clustering.txt.phrases", "Evaluation measures (information retrieval).txt.phrases", "Evaluation of binary classifiers.txt.phrases", "Extended Boolean model.txt.phrases", "Feature Selection Toolbox.txt.phrases", "Feature selection.txt.phrases", "Heaps' law.txt.phrases", "Hierarchical clustering.txt.phrases", "Information retrieval.txt.phrases", "K-means clustering.txt.phrases", "K-nearest neighbors algorithm.txt.phrases", "Language model.txt.phrases", "Lemmatisation.txt.phrases", "Multiclass classification.txt.phrases", "Naive Bayes classifier.txt.phrases", "Nearest centroid classifier.txt.phrases", "Nearest neighbor.txt.phrases", "Phrase search.txt.phrases", "Probabilistic relevance model.txt.phrases", "Query understanding.txt.phrases", "Relevance feedback.txt.phrases", "Search data structure.txt.phrases", "Search engine indexing.txt.phrases", "Skip list.txt.phrases", "Speech disorder.txt.phrases", "Spell checker.txt.phrases", "Standard Boolean model.txt.phrases", "Stemming.txt.phrases", "Stop words.txt.phrases", "Text segmentation.txt.phrases", "Tf-idf.txt.phrases", "Vector space model.txt.phrases", "Wildcard character.txt.phrases", "Zipf's law.txt.phrases"]

        for filename in os.listdir(ir_directory):
            if filename.endswith('phrases') and filename.startswith('iir'):


                filenamek = filename.split(".")[0].replace(" ","_")
                fileobj[filenamek] = np.zeros(200)
                fcount[filenamek]=0
                with open(ir_directory+filename,'r') as irfile:
                    # print(irfile)
                    for line in irfile.readlines():
                        line = line.split(",")[0]
                        # print(line)
                        concept = multiwordstem(' '.join(line.lower().split()))
                        # print (concept)
                        if(concept in dp.idx2prod):
                            # print("found",concept)
                            idx = dp.idx2prod.tolist().index(concept)
                            fcount[filenamek] += 1
                            fileobj[filenamek] += doc_embed[0][idx]
                        # else:
                            # print("not found",concept)
                # print(fileobj[filename])
                # print(fcount[filename])
                            # print(doc_embed[0][idx])

            if filename in wikifiles:
                filenamek = filename.split(".")[0].lower().replace("_"," ")
                wikiobj[filenamek] = np.zeros(200)
                fcount[filenamek] = 0
                with open(ir_directory+filename,'r') as irfile:
                    # print(irfile)
                    for line in irfile.readlines():
                        line = line.split(",")[0]

                        # print(line)
                        concept = ' '.join(line.lower().split())

                        if(concept in dp.idx2prod):
                            # print(" found ",concept)
                            fcount[filenamek] += 1
                            idx = dp.idx2prod.tolist().index(concept)
                            wikiobj[filenamek] += doc_embed[0][idx]
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
                docwikiscore[doc].put((spatial.distance.cosine(fileobj[doc],wikiobj[wiki]) ,wiki))


        # print(prediction)


        wikiannotations = '../../wikiannotationssmall_multiple.csv'
        file = open(wikiannotations,'r')
        import csv
        reader = csv.reader(file)
        true = {}
        for row in reader:
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