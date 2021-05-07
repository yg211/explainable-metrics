import os
from sts_models.psbert import BertSTSModel

class PSBertModel():
    def __init__(self, trained_model=None, bert_type='bert-large', checkpoint=False, batch_size=8, gpu=True):
        model_path = self._get_model_path(trained_model)
        self.sts_model = BertSTSModel(gpu=gpu,batch_size=batch_size,bert_type=bert_type,model_path=model_path, cp=checkpoint)

    def __call__(self, sent_pairs):
        scores = self.sts_model(sent_pairs)[0].reshape(1,-1)[0]
        return scores

    def _get_model_path(self, trained_model):

        link = 'https://www.dropbox.com/s/u9lziufawsxo7kz/bert_large-equal_newhans_sts.state_dict'
        cur_dir, _ = os.path.split(__file__)
        model_dir = os.path.join(cur_dir,'.sts_model')
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







