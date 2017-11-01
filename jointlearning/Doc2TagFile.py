import nltk
import csv
import sys
import os
import numpy as np
from concept_extractor import preprocessText,load_document
from concept_extractor import stem
import matplotlib.pyplot as plt
import plotly.plotly as py

trainin_text = 'data/keyphrase/textbook/all_text.csv'
conceptfile = 'data/conceptdocs.csv'

IR_CORPUS = 'data/keyphrase/textbook/all_text.csv'

## Extract Concepts
CONCEPT_FOLDER_BASE="data/keyphrase_output/"


def extractConcepts(outputfile,listbooks_concepts,preprocessed_concepts=[],notprocessed_concepts=[]):

    iCount = 0
    l_concepts = {}
    for pconceptfolders in preprocessed_concepts:
        for file in os.listdir(CONCEPT_FOLDER_BASE+pconceptfolders):
            # print(pconceptfolders)
            if file.endswith("phrases") and file.startswith(tuple(listbooks_concepts)):
                # print(file)
                fnamek = file.replace(".txt.phrases","")
                if(fnamek not in l_concepts):
                    l_concepts[fnamek] = set()

                for line in open(CONCEPT_FOLDER_BASE+pconceptfolders+"/"+file,'r').readlines():
                    if(len(line.strip()) > 1):
                        concept = line.split(",")[0].replace(" ","_")
                        tfidf = float(line.split(",")[1].strip())
                        if tfidf > 0.3:
                            l_concepts[fnamek].add(concept)

    for notpconceptfolder in notprocessed_concepts:
        for file in os.listdir(CONCEPT_FOLDER_BASE+notpconceptfolder):
            if file.endswith("phrases") and file.startswith(tuple(listbooks_concepts)):
                fnamek = file.replace(".txt.phrases","")
                if(fnamek not in l_concepts):
                    l_concepts[fnamek] = set()

                for line in open(CONCEPT_FOLDER_BASE+notpconceptfolder+"/"+file,'r').readlines():
                    pconcept = preprocessText(line.split(",")[0])
                    l_concepts[fnamek].add('_'.join(pconcept))

    documents = load_document(IR_CORPUS,listbooks_concepts)


    fcsv = open(outputfile,'w')
    for doc in documents:
        if doc.id  in l_concepts:
            # print(' '.join(l_concepts[doc.id]).replace("\n","").replace("\t","")+"\n")

            fcsv.write(doc.id.replace(" " ,"_")+" "+' '.join(l_concepts[doc.id]).replace("\n","").replace("\t","")+"\n")
            fcsv.write(' '.join(preprocessText(doc.text,stemming=False,stopwords_removal=False))+"\n")

    return l_concepts

def extractConcepts(outputfile,listbooks_concepts,preprocessed_concepts=[],notprocessed_concepts=[],tfidfscore=0.2):

    iCount = 0
    l_concepts = {}
    for pconceptfolders in preprocessed_concepts:
        for file in os.listdir(CONCEPT_FOLDER_BASE+pconceptfolders):
            # print(pconceptfolders)
            if file.endswith("phrases") and file.startswith(tuple(listbooks_concepts)):
                # print(file)
                fnamek = file.replace(".txt.phrases","")
                if(fnamek not in l_concepts):
                    l_concepts[fnamek] = set()

                for line in open(CONCEPT_FOLDER_BASE+pconceptfolders+"/"+file,'r').readlines():
                    if(len(line.strip()) > 1):
                        concept = line.split(",")[0].replace(" ","_")
                        tfidf = float(line.split(",")[1].strip())
                        if tfidf > tfidfscore:
                            l_concepts[fnamek].add(concept)

    for notpconceptfolder in notprocessed_concepts:
        for file in os.listdir(CONCEPT_FOLDER_BASE+notpconceptfolder):
            if file.endswith("phrases") and file.startswith(tuple(listbooks_concepts)):
                fnamek = file.replace(".txt.phrases","")
                if(fnamek not in l_concepts):
                    l_concepts[fnamek] = set()

                for line in open(CONCEPT_FOLDER_BASE+notpconceptfolder+"/"+file,'r').readlines():
                    pconcept = preprocessText(line.split(",")[0])
                    l_concepts[fnamek].add('_'.join(pconcept))

    documents = load_document(IR_CORPUS,listbooks_concepts)


    fcsv = open(outputfile,'w')
    lConcept_len = []
    for doc in documents:
        if doc.id not in l_concepts:
            l_concepts[doc.id] = []
        lConcept_len.append(len(l_concepts[doc.id]))
        fcsv.write(doc.id.replace(" " ,"_")+" "+' '.join(l_concepts[doc.id]).replace("\n","").replace("\t","")+"\n")
        fcsv.write(' '.join(preprocessText(doc.text,stemming=False,stopwords_removal=False))+"\n")

    return l_concepts,lConcept_len

def extractConceptst(outputfile,listbooks_concepts,preprocessed_concepts=[],notprocessed_concepts=[],topcount=1):

    iCount = 0
    l_concepts = {}
    for pconceptfolders in preprocessed_concepts:
        for file in os.listdir(CONCEPT_FOLDER_BASE+pconceptfolders):
            # print(pconceptfolders)
            if file.endswith("phrases") and file.startswith(tuple(listbooks_concepts)):
                # print(file)
                fnamek = file.replace(".txt.phrases","")
                if(fnamek not in l_concepts):
                    l_concepts[fnamek] = set()
                iCount = 0
                for line in open(CONCEPT_FOLDER_BASE+pconceptfolders+"/"+file,'r').readlines():
                    if(len(line.strip()) > 1):
                        concept = line.split(",")[0].replace(" ","_")
                        tfidf = float(line.split(",")[1].strip())
                        if iCount < topcount:
                            iCount += 1
                            l_concepts[fnamek].add(concept)

    for notpconceptfolder in notprocessed_concepts:
        for file in os.listdir(CONCEPT_FOLDER_BASE+notpconceptfolder):
            if file.endswith("phrases") and file.startswith(tuple(listbooks_concepts)):
                fnamek = file.replace(".txt.phrases","")
                if(fnamek not in l_concepts):
                    l_concepts[fnamek] = set()

                for line in open(CONCEPT_FOLDER_BASE+notpconceptfolder+"/"+file,'r').readlines():
                    pconcept = preprocessText(line.split(",")[0])
                    l_concepts[fnamek].add('_'.join(pconcept))

    documents = load_document(IR_CORPUS,listbooks_concepts)


    fcsv = open(outputfile,'w')
    lConcept_len = []
    for doc in documents:
        if doc.id not in l_concepts:
            l_concepts[doc.id] = []
        lConcept_len.append(len(l_concepts[doc.id]))
        fcsv.write(doc.id.replace(" " ,"_")+" "+' '.join(l_concepts[doc.id]).replace("\n","").replace("\t","")+"\n")
        fcsv.write(' '.join(preprocessText(doc.text,stemming=False,stopwords_removal=False))+"\n")

    return l_concepts,lConcept_len
if __name__ == '__main__':
    for tfidf in []:#["0.2","0.3","0.4","0.5"]:
        listbooks_concepts_train = ['irv-', 'issr-', 'foa-', 'sigir-', 'wiki-', 'zhai-']
        listbooks_concepts_test = [ 'mir-', 'iir-' ,'chapterwiseiir', 'iirbookpubs-','iirtest-','wikitest-']

        preprocessed_concepts = ['TFIDF1', 'TFIDF2', 'TFIDF3', 'TFIDFNP1', 'TFIDFNP2', 'TFIDFNP3']
        notprocessed_concepts = ["gold"]


        outputfile_train='doc2tagtrain_nostopwords_nostem.csv.'+tfidf+'tfidfnpg'
        outputfile_test = 'doc2tagtest_nostopwords_nostem.csv.'+tfidf+'tfidfnpg'
        extractConcepts(outputfile_train,listbooks_concepts_train,preprocessed_concepts,notprocessed_concepts,float(tfidf))
        extractConcepts(outputfile_test,listbooks_concepts_test,preprocessed_concepts,notprocessed_concepts,float(tfidf))


        preprocessed_concepts = ['TFIDFNP1', 'TFIDFNP2', 'TFIDFNP3']
        notprocessed_concepts = ["gold"]


        outputfile_train='doc2tagtrain_nostopwords_nostem.csv.'+tfidf+'ng'
        outputfile_test = 'doc2tagtest_nostopwords_nostem.csv.'+tfidf+'ng'
        extractConcepts(outputfile_train,listbooks_concepts_train,preprocessed_concepts,notprocessed_concepts,float(tfidf))
        extractConcepts(outputfile_test,listbooks_concepts_test,preprocessed_concepts,notprocessed_concepts,float(tfidf))

        preprocessed_concepts = ['TFIDFNP1', 'TFIDFNP2', 'TFIDFNP3']
        notprocessed_concepts = []


        outputfile_train='doc2tagtrain_nostopwords_nostem.csv.'+tfidf+'n'
        outputfile_test = 'doc2tagtest_nostopwords_nostem.csv.'+tfidf+'n'
        extractConcepts(outputfile_train,listbooks_concepts_train,preprocessed_concepts,notprocessed_concepts,float(tfidf))
        extractConcepts(outputfile_test,listbooks_concepts_test,preprocessed_concepts,notprocessed_concepts,float(tfidf))


    histogram = plt.figure()

    bins = np.linspace(0,15,16)

    for tfidf in []:#np.arange(0.25,3,0.01):#[0,0.1,0.2,0.3,0.4,0.5,0.6]:
        listbooks_concepts = ['irv-', 'issr-', 'foa-', 'sigir-', 'wiki-', 'zhai-','mir-', 'iir-', 'chapterwiseiir', 'iirbookpubs-', 'iirtest-', 'wikitest-']
        # listbooks_concepts_test = ['mir-', 'iir-', 'chapterwiseiir', 'iirbookpubs-', 'iirtest-', 'wikitest-']
        preprocessed_concepts = ['TFIDF1', 'TFIDF2', 'TFIDF3']
        notprocessed_concepts = []

        l_concepts, lConcept_len = extractConcepts("temps",listbooks_concepts,preprocessed_concepts,notprocessed_concepts,float(tfidf))
        setAll = set()
        for key in l_concepts:
            for con in l_concepts[key]:
                setAll.add(con)
        l_zeros = [con for con in lConcept_len if con == 0]
        print(tfidf,len(l_zeros),min(lConcept_len),max(lConcept_len),np.mean(lConcept_len),np.median(lConcept_len),len(setAll))

    for topcount in range(1,2):
        print("hi")
        listbooks_concepts = ['irv-', 'issr-', 'foa-', 'sigir-', 'wiki-', 'zhai-', 'mir-', 'iir-', 'chapterwiseiir',
                              'iirbookpubs-', 'iirtest-', 'wikitest-']
        # listbooks_concepts_test = ['mir-', 'iir-', 'chapterwiseiir', 'iirbookpubs-', 'iirtest-', 'wikitest-']
        preprocessed_concepts = ['TFIDF1', 'TFIDF2', 'TFIDF3']
        preprocessed_concepts = ['TFIDF1', 'TFIDF2', 'TFIDF3', 'TFIDFNP1', 'TFIDFNP2', 'TFIDFNP3']

        notprocessed_concepts = []

        l_concepts, lConcept_len = extractConceptst("temps", listbooks_concepts, preprocessed_concepts,
                                                   notprocessed_concepts, topcount)
        setAll = set()
        for key in l_concepts:
            for con in l_concepts[key]:
                setAll.add(con)
        l_zeros = [con for con in lConcept_len if con == 0]
        print(topcount, len(l_zeros), min(lConcept_len), max(lConcept_len), np.mean(lConcept_len),
              np.median(lConcept_len), len(setAll))

        #     plt.hist(lConcept_len, bins)
#
# plt.show()

    # plot_url = py.plot_mpl(histogram, filename='docs/histogram-mpl-same')



