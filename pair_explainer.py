import sys
sys.path.append('../')

from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import shap
from bert_sim import BertSTSModel
import os
import shutil
import subprocess

class STSWrapper():
    def __init__(self, trained_model, bert_type, checkpoint, batch_size, gpu, tokenizer='nltk'):
        self.sts_model = BertSTSModel(gpu=gpu,batch_size=batch_size,bert_type=bert_type,model_path=trained_model, cp=checkpoint)
        if tokenizer == 'nltk': self.tokenizer = word_tokenize
        elif tokenizer == 'split': self.tokenizer = self._split_tokenizer
            
    def _split_tokenizer(self, sent):
        return sent.split()

    def __call__(self, sent_pair_list):
        batch = []
        for pair in sent_pair_list:
            s1,s2 = pair[0].split('[SEP]')
            batch.append( (s1,s2) )

        scores = self.sts_model(batch)[0].reshape(1,-1)[0]
        return scores

    def _tokenize_sent(self, sentence):
        if isinstance(sentence,str):
            #token_ids = self.sts_model.tokenizer.encode(sentence)
            #tokens = self.sts_model.tokenizer.convert_ids_to_tokens(token_ids)[1:-1]
            tokens = self.tokenizer(sentence)
        elif isinstance(sentence, list):
            tokens = sentence

        return tokens

    def build_feature(self, sent1, sent2):
        tokens1 = self._tokenize_sent(sent1)
        tokens2 = self._tokenize_sent(sent2)
        self.s1len = len(tokens1)
        self.s2len = len(tokens2)

        tdict = {}
        for i in range(len(tokens1)):
            tdict['s1_{}'.format(i)] = tokens1[i]
        for i in range(len(tokens2)):
            tdict['s2_{}'.format(i)] = tokens2[i]

        return pd.DataFrame(tdict, index=[0])

    def mask_model(self, mask, x):
        tokens = []
        for mm, tt in zip(mask, x):
            if mm: tokens.append(tt)
            else: tokens.append('[MASK]')
        s1 = ' '.join(tokens[:self.s1len])
        s2 = ' '.join(tokens[self.s1len:])
        sentence_pair = pd.DataFrame([s1+'[SEP]'+s2])
        return sentence_pair



class ExplainableSTS():
    def __init__(self, trained_model=None, bert_type='bert-large', checkpoint=False, batch_size=8, gpu=True):
        model_path = self.get_model_path(trained_model)
        self.wrapper = STSWrapper(model_path, bert_type, checkpoint, batch_size, gpu)

    def __call__(self, sent1, sent2):
        s1 = ' '.join(self.wrapper.tokenizer(sent1))
        s2 = ' '.join(self.wrapper.tokenizer(sent2))
        return self.wrapper.sts_model([(s1,s2)])[0].reshape(1,-1)[0][0]

    def explain(self, sent1, sent2, plot=False):
        explainer = shap.Explainer(self.wrapper, self.wrapper.mask_model)
        value = explainer(self.wrapper.build_feature(sent1, sent2))
        if plot: shap.plots.waterfall(value[0])
        all_tokens = [] 
        all_tokens += ['s1_'+t for t in self.wrapper.tokenizer(sent1)] 
        all_tokens += ['s2_'+t for t in self.wrapper.tokenizer(sent2)] 

        return [(token,sv) for token, sv in zip(all_tokens,value[0].values)]

    def get_model_path(self, trained_model):

        link = 'https://www.dropbox.com/s/u9lziufawsxo7kz/bert_large-equal_newhans_sts.state_dict'
        model_dir = './.sts_model/'
        default_model_path = os.path.join(model_dir,'bert_large-equal_newhans_sts.state_dict')


        if trained_model is None:
            if os.path.isfile(default_model_path): return default_model_path
            else:
                if not os.path.exists(model_dir): os.makedirs(model_dir)
                process = subprocess.Popen(['wget','-O', default_model_path, link], stdout=subprocess.PIPE)
                output, error = process.communicate()
                return default_model_path
        else:
            assert os.path.isfile(trained_model) 
            return trained_model









