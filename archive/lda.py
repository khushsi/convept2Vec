import os

from w2v_base import  W2V_base
from gensim.models import LdaModel
import json

class W2V_cpp2(W2V_base):
    def __init__(self,n_topic, path_review, path_business, folder):
        self.n_topic = n_topic

        print('Init W2V_base')
        W2V_base.__init__(self, path_review, path_business, folder)

        print('Process idx2prod')
        #process dict
        for prod_id in self.idx2prod.keys():
            prod = self.idx2prod[prod_id]
            n_prod_id = prod_id - len(self.word_count) - 1
            del self.idx2prod[prod_id]
            self.idx2prod[n_prod_id] = prod
            self.prod2idx[prod] = n_prod_id

        print('Process idx2user')
        for user_id in self.idx2user.keys():
            user = self.idx2user[user_id]
            n_user_id = user_id - len(self.word_count) - len(self.prod2idx) - 1
            del self.idx2user[user_id]
            self.idx2user[n_user_id] = user
            self.user2idx[user] = n_user_id

        print('Init W2V_cpp2 done')


    def train(self):
        data = []
        entity2id = {}
        id2entity = []

        print('Loading data')
        for obj in self.data:
            doc = []
            obj_sents = obj["text_data"]
            entity = obj["prod"]
            if entity not in entity2id:
                entity2id[entity] = len(entity2id)
                id2entity.append(entity)
            doc_id = entity2id[entity]

            for obj_sent in obj_sents:
                for pair in obj_sent:
                    if pair[0] >= 0:
                        doc.append((pair[0], doc_id))
            data.append(doc)

        print('Start training')
        self.ldamodel = LdaModel(corpus=data, id2word=self.idx2word, num_topics=self.n_topic)
        print('Training complete, start exporting')

        f_entity = open(home + "/Data/yelp/lda/prod.txt", "w")
        f_model = open(home + "/Data/yelp/lda/model.txt", "w")
        f_model.write(str(len(entity2id)))
        f_model.write(" ")
        f_model.write(str(self.n_topic))
        f_model.write("\n")

        for entity in id2entity:
            f_entity.write(entity)
            f_entity.write("\n")

            f_model.write(entity)
            f_model.write(" ")

            distr = self.ldamodel.get_document_topics(data[1], minimum_phi_value=0, minimum_probability=0)
            distr = [pair[1] for pair in distr]

            for prod in distr:
                f_model.write(str(prod))
                f_model.write(" ")

            f_model.write("\n")

        self.ldamodel.save(home + "/Data/yelp/lda/model_200")


if __name__ == '__main__':
    home = os.environ["HOME"]
    lda = W2V_cpp2(200, home + "/Data/yelp/output/raw_review_restaurant.json", home + "/Data/yelp/yelp_academic_dataset_business.json", "yelp_rest_prod")
    print("init")
    lda.train()
    print("done")