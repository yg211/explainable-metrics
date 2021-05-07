

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import *
import torch


class BertScoreModel():
    def __init__(self,metric='f1',gpu=True):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.model = BertModel.from_pretrained('bert-large-uncased')
        self.gpu = gpu
        self.metric = metric.lower().strip()

        if self.gpu:
            self.model.to('cuda')

        assert self.metric in ['f1', 'recall', 'precision']


    def get_token_vecs(self,sent):
        tokens = self.tokenizer.encode(sent)
        tokens = torch.tensor(tokens).unsqueeze(0)
        if self.gpu:
            tokens = tokens.to('cuda')
        vecs = self.model(tokens).last_hidden_state[0].data.cpu().numpy()
        return vecs


    def get_sim_metric(self, sent1_vecs, sent2_vecs):
        sim_matrix = cosine_similarity(sent2_vecs,sent1_vecs)
        
        recall = np.mean(np.max(sim_matrix,axis=1))
        precision = np.mean(np.max(sim_matrix,axis=0))
        if recall+precision == 0:
            f1 = None
        else:
            f1 = 2.*recall*precision/(recall+precision)

        if self.metric == 'f1': return f1
        elif self.metric == 'recall': return recall
        elif self.metric == 'precision': return precision

        
    def __call__(self, sent_pairs):
        scores = []
        for (s1,s2) in sent_pairs: 
            vec1 = self.get_token_vecs(s1)
            vec2 = self.get_token_vecs(s2)
            ss = self.get_sim_metric(vec1, vec2)
            scores.append(ss)
        return np.array(scores)




if __name__ == '__main__':
    s1 = 'Tom beats Bob and wins the game.'
    s2 = 'Bob defeats Tom and wins the competition.'

    model = BertScoreWrapper()
    score = model([(s1, s2)])
    print(score)





