import nltk
import csv
import sys
import os
from concept_extractor import preprocessText,load_document
from concept_extractor import stem

trainin_text = 'data/keyphrase/textbook/all_text.csv'
conceptfile = 'data/conceptdocs.csv'



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

listbooks_concepts = [ 'mir-', 'iir-' ,'chapterwiseiir-','wikitest-', 'iirbookpubs-','iirtest-']
preprocessed_concepts=['TFIDF1','TFIDF2','TFIDF3','TFIDFNP1','TFIDFNP2','TFIDFNP3']
notprocessed_concepts=["gold"]


iCount = 0
l_concepts = set()
for pconceptfolders in preprocessed_concepts:
    for file in os.listdir(CONCEPT_FOLDER_BASE+pconceptfolders):
        if file.endswith("phrases") and file.startswith(tuple(listbooks_concepts)):
            for line in open(CONCEPT_FOLDER_BASE+pconceptfolders+"/"+file,'r').readlines():
                concept = line.split(",")[0]
                l_concepts.add(concept)


for notpconceptfolder in notprocessed_concepts:
    for file in os.listdir(CONCEPT_FOLDER_BASE+notpconceptfolder):
        fwrite = open(CONCEPT_FOLDER_BASE+notpconceptfolder+"_stem/"+file,'w')
        if file.endswith("phrases") and file.startswith(tuple(listbooks_concepts)):
            for line in open(CONCEPT_FOLDER_BASE+notpconceptfolder+"/"+file,'r').readlines():
                pconcept = preprocessText(line.split(",")[0])
                fwrite.write(' '.join(pconcept)+"\n")
                l_concepts.add(' '.join(pconcept))


fconcepts = ['data/file2.txt','data/file3.txt']
## From Files

for cfile in fconcepts:
    for line in open(cfile).readlines():
        l_concepts.add(line.strip())


category_file = 'data/chapterwise_title.csv'
chapterwise_titledict = conceptcategories(category_file)

listbooks = ['irv-','issr-','foa-','sigir-','zhai-','seirip-','wiki-']
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
    fw.write(key+","+' '.join(conceptdocs[key]).replace(","," ")+ "\n")

f_review = 'data/yelp_academic_dataset_review.json'
f_business = 'data/yelp_academic_dataset_business.json'
f_doc2tagwithtitle = 'data/doc2tag_title.json'

reviewwriter = open(f_review,'w')
businesswriter = open(f_business,'w')
doc2tagwirter = open(f_doc2tagwithtitle,'w')
for key in conceptdocs.keys():
    categories = ""
    if(key in conceptcategory):
        categories = "','".join(list(set(conceptcategory[key])))
        cat = [con.replace(" ","_") for con in conceptcategory[key]]
    # print(key, " - " ,categories)
    rjsontext = "{\"review_id\": \"r"+key+"\", \"user_id\": \"uiir_1\",\"business_id\":\""+key+"\" , \"stars\": 5, \"date\": \"2011-10-10\", \"text\": \""+' '.join(conceptdocs[key])+"\"}"+"\n"
    bjsontext = "{\"business_id\":\""+key+"\",\"name\":\""+key+"\",\"categories\":\"['"+categories+"']\",\"type\":\"business\"}" + "\n"

    reviewwriter.write(rjsontext)
    businesswriter.write(bjsontext)

    doc2tagwirter.write(key.replace(" ", "_") + " " + ' '.join(cat).replace("\n", "").replace("\t", "") + "\n")
    doc2tagwirter.write(' '.join(conceptdocs[key]) + "\n")




