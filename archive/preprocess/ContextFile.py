import nltk
import csv
import sys
import os
from concept_extractor import preprocessText,load_document
from concept_extractor import stem

trainin_text = 'data/keyphrase/textbook/all_text.csv'
conceptfile = 'data/conceptdocs.csv'
wikifiles = ['Information retrieval','Accuracy and precision','Adversarial information retrieval','Association for Computing Machinery','Bayes\' theorem','Binary Independence Model','Binary classification','Categorization','Center for Intelligent Information Retrieval','Classification of the sciences (Peirce)','Co-occurrence','Collaborative information seeking','Computational linguistics','Computer data storage','Conference on Information and Knowledge Management','Confusion matrix','Controlled vocabulary',
             'Cross-language information retrieval','Data mining','Data modeling','Dimensionality reduction',
             'Discounted cumulative gain','Human-computer information retrieval','Conference on Human Factors in Computing Systems',
             'Cluster analysis','Data compression','Human-computer interaction','Information visualization','Search engine indexing',
             'Web search engine','Information retrieval','Human-computer information retrieval','Lemmatisation','Stemming',
             'Vector space model','Generalized vector space model','Bias-variance tradeoff','Supervised learning','Data compression',
             'Controlled vocabulary','Vocabulary','Data pre-processing','Electronic data processing','Document classification',
             'Classification','Document clustering','Hierarchical clustering','Evaluation measures (information retrieval)','Evaluation of binary classifiers',
             'Binary classification','Extended Boolean model','Fuzzy retrieval','Feature selection','Feature Selection Toolbox','Heaps\' law',
             'K-means clustering','K-nearest neighbors algorithm','Nearest neighbor','Language model','Modeling language',
             'Multiclass classification','Statistical classification','Naive Bayes classifier','Nearest centroid classifier','Phrase search',
             'Phrase','Probabilistic relevance model','Okapi BM25','Query understanding','Web query classification','Relevance feedback',
             'Search data structure','Persistent data structure','Skip list','List of Skip Beat! chapters','Speech disorder','Communication disorder',
             'Spell checker','Grammar checker','Standard Boolean model','Stop words','Text segmentation','Wildcard character','Text normalization',
             'Zipf\'s law','Tf-idf']

wikifiles = [''.join(preprocessText(wiki)) for wiki in wikifiles]


def conceptcategories(category_file):
    filecsv = csv.reader(open(category_file))
    categorydict = {}
    next(filecsv)
    for row in filecsv:
        categorydict[row[0].strip()] = row[1].strip()

    return categorydict

def filltokendict(document,category=None):

    doc = document.text

    tokens = preprocessText(doc,stemming=False,stopwords_removal=True)

    doc_concepts= set()
    ngrams = nltk.ngrams(tokens,n=6)
    for ngram in ngrams:
        token=stem(ngram[0])
        # print(ngram[0])

        if token in l_concepts:
            if token in conceptdocs:
                conceptdocs[token] += ngram[1:]
            else:
                conceptdocs[token] = list(ngram[1:])
            doc_concepts.add(token)


        token=stem(ngram[5])
        if token in l_concepts:
            if token in conceptdocs:
                conceptdocs[token] += ngram[:5]
                # print(ngram[:4])
            else:
                conceptdocs[token] = list(ngram[:5])
            doc_concepts.add(token)


    ngrams = nltk.ngrams(tokens, n=7)
    for ngram in ngrams:

        token=' '.join([stem(ngram[5]),stem(ngram[6])])
        # print(token)
        if token in l_concepts:
            # print(token)
            if token in conceptdocs:
                conceptdocs[token] += ngram[:5]
            else:
                conceptdocs[token] = list(ngram[:5])
            # print(conceptdocs[token])
            doc_concepts.add(token)

        token=' '.join([stem(ngram[0]),stem(ngram[1])])
        if token in l_concepts:
            # print(token)
            if token in conceptdocs:
                conceptdocs[token] += ngram[2:]
            else:
                conceptdocs[token] = list(ngram[2:])
            # print(conceptdocs[token])
            doc_concepts.add(token)

    ngrams = nltk.ngrams(tokens, n=8)
    for ngram in ngrams:

        token=' '.join([stem(ngram[5]),stem(ngram[6]),stem(ngram[7])])
        # print(token)
        if token in l_concepts:
            # print(token)
            if token in conceptdocs:
                conceptdocs[token] += ngram[:5]
            else:
                conceptdocs[token] = list(ngram[:5])
            # print(conceptdocs[token])
            doc_concepts.add(token)

        token=' '.join([stem(ngram[0]),stem(ngram[1]),stem(ngram[2])])
        if token in l_concepts:

            if token in conceptdocs:
                conceptdocs[token] += ngram[3:]

            else:
                conceptdocs[token] = list(ngram[3:])
            # print(conceptdocs[token])
            doc_concepts.add(token)

    if(category != None and category != ""):
        for concept in doc_concepts:
            if(concept in conceptcategory):
                conceptcategory[concept].append(category)
            else:
                conceptcategory[concept] = [category]

conceptdocs={}
conceptcategory = {}

## Extract Concepts
CONCEPT_FOLDER_BASE="data/keyphrase_output/"

listbooks_concepts = [ 'mir-', 'iir-' ,'chapterwiseiir','wiki', 'iirbookpubs-']
preprocessed_concepts=['TFIDF1','TFIDF2','TFIDF3','TFIDFNP1','TFIDFNP2','TFIDFNP3']
notprocessed_concepts=["gold"]

iCount = 0
l_concepts = set()
for pconceptfolders in preprocessed_concepts:
    for file in os.listdir(CONCEPT_FOLDER_BASE+pconceptfolders):
        if file.endswith("phrases") and file.startswith(tuple(listbooks_concepts)):
            file = file.replace("wikitest-", "wiki-")
            if(file.startswith("wiki-")):
                filekk = ''.join(preprocessText(file.replace(".txt.phrases","").replace("wiki-","")))
                if (filekk in wikifiles):
                    for line in open(CONCEPT_FOLDER_BASE + pconceptfolders + "/" + file, 'r').readlines():
                        concept = line.split(",")[0]
                        l_concepts.add(concept)
            else:
                for line in open(CONCEPT_FOLDER_BASE+pconceptfolders+"/"+file,'r').readlines():
                    concept = line.split(",")[0]
                    l_concepts.add(concept)

for notpconceptfolder in notprocessed_concepts:
    for file in os.listdir(CONCEPT_FOLDER_BASE+notpconceptfolder):
        if file.endswith("phrases") and file.startswith(tuple(listbooks_concepts)):
            for line in open(CONCEPT_FOLDER_BASE+notpconceptfolder+"/"+file,'r').readlines():
                pconcept = preprocessText(line.split(",")[0])
                l_concepts.add(' '.join(pconcept))


fconcepts = ['data/file2.txt','data/file3.txt']
## From Files

for cfile in fconcepts:
    for line in open(cfile).readlines():
        l_concepts.add(line.strip())

category_file = 'data/chapterwise_title.csv'
chapterwise_titledict = conceptcategories(category_file)

listbooks = ['irv','issr','mir','iir-','foa','sigir','zhai','iirbookpubs','seirip','chapterwiseiir','wiki']
# listbooks = ['sigir']
documents = load_document(trainin_text,booknames=listbooks)

for doc in documents:
    if(doc.id in chapterwise_titledict):
        filltokendict(doc,chapterwise_titledict[doc.id])
    else:
        filltokendict(doc)


# No of Concepts
print("No of Concepts :",len(conceptdocs))

fw = open(conceptfile,"w")

for key in conceptdocs:
    fw.write(key+"\t"+' '.join(conceptdocs[key])+ "\n")

f_review = 'data/yelp_academic_dataset_review.json'
f_business = 'data/yelp_academic_dataset_business.json'
reviewwriter = open(f_review,'w')
businesswriter = open(f_business,'w')

for key in conceptdocs.keys():
    categories = ""
    if(key in conceptcategory):
        categories = "','".join(list(set(conceptcategory[key])))
    # print(key, " - " ,categories)
    rjsontext = "{\"review_id\": \"r"+key+"\", \"user_id\": \"uiir_1\",\"business_id\":\""+key+"\" , \"stars\": 5, \"date\": \"2011-10-10\", \"text\": \""+' '.join(conceptdocs[key])+"\"}"+"\n"
    bjsontext = "{\"business_id\":\""+key+"\",\"name\":\""+key+"\",\"categories\":\"['"+categories+"']\",\"type\":\"business\"}" + "\n"

    reviewwriter.write(rjsontext)
    businesswriter.write(bjsontext)

# for concept in l_concepts:
#     if concept not in conceptdocs:
#         print(concept)