from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import shap
from bert_sim import BertSTSModel
import os
import shutil
import subprocess

class STSWrapper():
    def __init__(self, trained_model, ref_sent, bert_type, checkpoint, batch_size, gpu, tokenizer='nltk'):
        self.sts_model = BertSTSModel(gpu=gpu,batch_size=batch_size,bert_type=bert_type,model_path=trained_model, cp=checkpoint)
        self.ref_sent = ref_sent
        if tokenizer == 'nltk': self.tokenizer = word_tokenize

    def __call__(self, sent_list):
        batch = []
        for ss in sent_list:
            batch.append((' '.join(ss), self.ref_sent))

        scores = self.sts_model(batch)[0].reshape(1,-1)[0]
        return scores

    def build_feature(self, sentence):
        if isinstance(sentence,str):
            #token_ids = self.sts_model.tokenizer.encode(sentence)
            #tokens = self.sts_model.tokenizer.convert_ids_to_tokens(token_ids)[1:-1]
            tokens = self.tokenizer(sentence)
        elif isinstance(sentence, list):
            tokens = sentence
        tdict = {}
        for i in range(len(tokens)):
            tdict['{}'.format(i)] = tokens[i]
        return pd.DataFrame(tdict, index=[0])

    def mask_model(self, mask, x):
        sent = []
        for mm, tt in zip(mask, x):
            if mm: sent.append(tt)
            else: sent.append('[MASK]')
        sentence = ' '.join(sent)
        sentence = pd.DataFrame([sentence])
        return sentence



class ExplainableSTS():
    def __init__(self, trained_model=None, ref_sent='', bert_type='bert-large', checkpoint=False, batch_size=8, gpu=True):
        model_path = self.get_model_path(trained_model)
        self.wrapper = STSWrapper(model_path, ref_sent, bert_type, checkpoint, batch_size, gpu)

    def __call__(sent1, sent2):
        return self.wrapper.sts_model([(sent1,sent2)])[0].reshape(1,-1)[0]

    def explain(self, sent1, sent2, plot=False):
        self.wrapper.ref_sent = sent1
        explainer = shap.Explainer(self.wrapper, self.wrapper.mask_model)
        value = explainer(self.wrapper.build_feature(sent2))
        if plot: shap.plots.waterfall(value[0])

        return [(token,sv) for token, sv in zip(self.wrapper.tokenizer(sent2),value[0].values)]

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









