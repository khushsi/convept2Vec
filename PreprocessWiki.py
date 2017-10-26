f = open('wikiannotations.csv','r', encoding='utf-8')
fw=open('/Users/khushsi/Data/yelp/yelp_academic_dataset_business.json','w')

fu=open('/Users/khushsi/Data/yelp/yelp_academic_dataset_user.json','w')
fr = open('/Users/khushsi/Data/yelp/yelp_academic_dataset_review.json','w')
import csv
import re

invalid_escape = re.compile(r'\\[0-7]{1,3}')  # up to 3 digits for byte values up to FF

def replace_with_byte(match):
    return chr(int(match.group(0)[1:], 8))

def repair(brokenjson):
    return invalid_escape.sub(replace_with_byte, brokenjson)

reader = csv.reader(f,delimiter=",")
for row in reader:
    print (row[0])
    print (row[1])
    print(row[2])

    wstrb="{\"business_id\":\""+row[0]+"\",\"name\":\""+row[1]+"\",\"categories\":[\""+row[2].replace(",","\",\"")+"\"],\"type\":\"business\"}"

    fw.write(wstrb+"\n")

    wstru="{\"user_id\":\"u"+row[0]+"\",\"name\":\"Rob\",\"type\":\"user\"}"

    fu.write(wstru+"\n")

    wstrr="{\"review_id\": \"r"+row[0]+"\", \"user_id\": \"uiir_1\",\"business_id\":\""+row[0]+"\" , \"stars\": 5, \"date\": \"2011-10-10\", \"text\": \""+repair(row[3].replace("\"","").replace("\,","").replace("\\",""))+"\", \"useful\": 0, \"funny\": 0,\"cool\": 0, \"type\": \"review\"}"
    fr.write(wstrr+"\n")

