import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from transformers import *
from utils.utils import build_batch

def get_bert_sdict(sdict):
    new_dict = {}
    for nn in sdict:
        # print(nn)
        if 'nli_head' not in nn: new_dict[nn[5:]] = sdict[nn]
    return new_dict


class BertSTSModel(nn.Module):
    def __init__(self, model_path=None, gpu=True, bert_type='bert-base', batch_size=8, cp=False):
        super(BertSTSModel, self).__init__()
        self.bert_type = bert_type
        self.checkpoint = cp 

        if 'bert-base' in bert_type:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif 'bert-large' in bert_type:
            self.bert = BertModel.from_pretrained('bert-large-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        elif 'albert' in bert_type:
            self.bert = AlbertModel.from_pretrained(bert_type)
            self.tokenizer = AlbertTokenizer.from_pretrained(bert_type)
        else:
            print('illegal bert type {}!'.format(bert_type))

        self.num_hidden_layers = self.bert.config.num_hidden_layers
        self.vdim = self.bert.config.hidden_size
        self.sts_head = nn.Linear(self.vdim,1)
        self.gpu = gpu
        self.batch_size=batch_size
        #self.bert.config.output_attentions=True
        #self.bert.config.output_hidden_states=True

        # load trained model
        if model_path is not None:
            if isinstance(model_path, OrderedDict):
                self.load_state_dict(model_path, strict=False)
            else:
                assert isinstance(model_path, str)
                if 'sts' in model_path:
                    sdict = torch.load(model_path,map_location=lambda storage, loc: storage)
                    self.load_state_dict(sdict, strict=False)
                else:
                    sdict = torch.load(model_path, map_location=lambda storage, loc: storage)
                    bert_sdict = get_bert_sdict(sdict)
                    self.bert.load_state_dict(bert_sdict, strict=False)

        if gpu:
            self.to('cuda')
            # torch.cuda.set_device(0)


    def load_model(self, sdict):
        if self.gpu:
            self.load_state_dict(sdict)
            self.to('cuda')
        else:
            self.load_state_dict(sdict)

    def forward(self, sent_pair_list, bs=None, output_attentions=False, output_hidden_states=False):
        all_scores = None
        all_hidden_states = None
        all_attentions = None

        if bs is None: 
            bs = self.batch_size
            no_prog_bar = True
        else: no_prog_bar = False
        for batch_idx in tqdm(range(0,len(sent_pair_list),bs), disable=no_prog_bar,desc='evaluate'):
            outputs = self.ff(sent_pair_list[batch_idx:batch_idx+bs],output_attentions,output_hidden_states)
            scores = outputs[0].data.cpu().numpy()
            if all_scores is None: all_scores = scores
            else: all_scores = np.append(all_scores , scores, axis=0)
            #
            if output_hidden_states: 
                hidden_states = outputs[1]
                assert hidden_states is not None
                if all_hidden_states is None: all_hidden_states = list(hidden_states)
                else: 
                    for i in range(len(all_hidden_states)):
                        all_hidden_states[i] = torch.cat((all_hidden_states[i], hidden_states[i]), 0)
            if output_attentions: 
                attentions = outputs[-1]
                assert attentions is not None
                if all_attentions is None: all_attentions = list(attentions)
                else: 
                    for i in range(len(all_attentions)):
                        all_attentions[i] = torch.cat((all_attentions[i], attentions[i]), 0)

        final_outputs = [all_scores]
        if output_hidden_states: final_outputs.append(all_hidden_states)
        if output_attentions: final_outputs.append(all_attentions)
        return tuple(final_outputs)

    def step_bert_encode(self, module, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(module.layer):
            if module.config.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = checkpoint.checkpoint(layer_module, hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if module.config.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if module.config.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if module.config.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if module.config.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


    def step_checkpoint_bert(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        modules = [module for k, module in self.bert._modules.items()]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to float if need + fp16 compatibility
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = modules[0](input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.step_bert_encode(modules[1], embedding_output,extended_attention_mask,head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = modules[2](sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


    def ff(self,sent_pair_list,output_attentions=False,output_hidden_states=False):
        ids, types, masks = build_batch(self.tokenizer, sent_pair_list, self.bert_type)
        if ids is None: return None
        ids_tensor = torch.tensor(ids) 
        types_tensor = torch.tensor(types) 
        masks_tensor = torch.tensor(masks) 
        
        if self.gpu:
            ids_tensor = ids_tensor.to('cuda')
            types_tensor = types_tensor.to('cuda')
            masks_tensor = masks_tensor.to('cuda')

        if self.checkpoint:
            outputs = self.step_checkpoint_bert(input_ids=ids_tensor, token_type_ids=types_tensor, attention_mask=masks_tensor) 
        else:
            outputs = self.bert(input_ids=ids_tensor, token_type_ids=types_tensor, attention_mask=masks_tensor, output_attentions=output_attentions, output_hidden_states=output_hidden_states) 

        cls_vecs = outputs[1]
        scores = self.sts_head(cls_vecs)
        scores = nn.Sigmoid()(scores/10.)

        attentions, hidden_states = None, None
        if output_attentions: attentions = outputs[-1]
        if output_hidden_states: hidden_states = outputs[2]

        '''
        TEST!!!! Finding: output of hidden_states: embd, layer0, ..., last layer (i.e. closest to the output)
        last_hidden = outputs[0]
        for i in range(len(hidden_states)):
            s = torch.sum(last_hidden- hidden_states[i])
            print('\n===>>  hidden idx', i, s)
        TEST!!!!
        '''
        final_outputs = [scores]
        if output_hidden_states: final_outputs.append(hidden_states)
        if output_attentions: final_outputs.append(attentions)
        return tuple(final_outputs)

    def save(self, output_path, config_dic=None, rho=None):
        if rho is None:
            model_name = 'sts_model.state_dict'
        else:
            model_name = 'sts_model_rho{}.state_dict'.format(rho)
        opath = os.path.join(output_path, model_name)
        if config_dic is None:
            torch.save(self.state_dict(),opath)
        else:
            torch.save(config_dic,opath)


if __name__ == '__main__':
    model = BertSTSModel()

    for pn, pp in model.named_parameters():
        print(pn, pp)
