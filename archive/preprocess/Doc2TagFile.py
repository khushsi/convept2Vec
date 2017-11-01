import nltk
import csv
import sys
import os
from concept_extractor import preprocessText,load_document
from concept_extractor import stem

trainin_text = 'data/keyphrase/textbook/all_text.csv'
conceptfile = 'data/conceptdocs.csv'



## Extract Concepts
CONCEPT_FOLDER_BASE="data/keyphrase_output/"

#listbooks_concepts = [ 'mir-', 'iir-' ,'chapterwiseiir','wiki', 'iirbookpubs-','iirtest-','irv-','issr-','foa-','sigir-']
listbooks_concepts = [ 'irv-','issr-','foa-','sigir-','wiki-','zhai-']
# listbooks_concepts = [ 'mir-', 'iir-' ,'chapterwiseiir', 'iirbookpubs-','iirtest-','wikitest-']
preprocessed_concepts=['TFIDF1','TFIDF2','TFIDF3','TFIDFNP1','TFIDFNP2','TFIDFNP3']
notprocessed_concepts=["gold"]


iCount = 0
l_concepts = {}
for pconceptfolders in preprocessed_concepts:
    for file in os.listdir(CONCEPT_FOLDER_BASE+pconceptfolders):
        if file.endswith("phrases") and file.startswith(tuple(listbooks_concepts)):
            fnamek = file.replace(".txt.phrases","")
            if(fnamek not in l_concepts):
                l_concepts[fnamek] = set()

            for line in open(CONCEPT_FOLDER_BASE+pconceptfolders+"/"+file,'r').readlines():
                concept = line.split(",")[0].replace(" ","_")
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

IR_CORPUS = 'data/keyphrase/textbook/all_text.csv'

documents = load_document(IR_CORPUS,listbooks_concepts)


outputfile='doc2tagtrain_nostopwords_nostem.csv'
fcsv = open(outputfile,'w')
for doc in documents:
    if doc.id  in l_concepts:
        # print(' '.join(l_concepts[doc.id]).replace("\n","").replace("\t","")+"\n")

        fcsv.write(doc.id.replace(" " ,"_")+" "+' '.join(l_concepts[doc.id]).replace("\n","").replace("\t","")+"\n")
        fcsv.write(' '.join(preprocessText(doc.text,stemming=False,stopwords_removal=False))+"\n")