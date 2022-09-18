'''
Description     : VisualBERT Localized-Answering model
Paper           : Surgical-VQLA: Transformer with Gated Vision-Language Embedding for 
                  Visual Question Localized-Answering in Robotic Surgery
Author          : Long Bai, Mobarakol Islam, Lalithkumar Seenivasan, Hongliang Ren
Lab             : Medical Mechatronics Lab, The Chinese University of Hong Kong
Acknowledgement : Code adopted from the official implementation of VisualBERT model 
                  from huggingface/transformers (https://github.com/huggingface/transformers.git).
'''

import torch
from torch import nn
from transformers import VisualBertModel, VisualBertConfig
import torch.nn.functional as F
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualBertPrediction(nn.Module):
    '''
    VisualBert Classification Model
    vocab_size    = tokenizer length
    encoder_layer = 6
    n_heads       = 8
    num_class     = number of class in dataset
    '''
    def __init__(self, vocab_size, layers, n_heads, num_class = 10):
        super(VisualBertPrediction, self).__init__()
        VBconfig = VisualBertConfig(vocab_size = vocab_size, visual_embedding_dim = 512, num_hidden_layers = layers, num_attention_heads = n_heads, hidden_size = 2048)
        self.VisualBertEncoder = VisualBertModel(VBconfig)
        self.classifier = nn.Linear(VBconfig.hidden_size, num_class)
        self.bbox_embed = MLP(2048, 2048, 4, 3)

    def forward(self, inputs, visual_embeds):
        # prepare visual embedding
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).to(device)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float).to(device)

        # append visual features to text
        inputs.update({
                        "visual_embeds": visual_embeds,
                        "visual_token_type_ids": visual_token_type_ids,
                        "visual_attention_mask": visual_attention_mask,
                        "output_attentions": True
                        })
        
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        inputs['visual_token_type_ids'] = inputs['visual_token_type_ids'].to(device)
        inputs['visual_attention_mask'] = inputs['visual_attention_mask'].to(device)

        # Encoder output
        outputs = self.VisualBertEncoder(**inputs)
        
        # classification layer
        classification_outputs = self.classifier(outputs['pooler_output'])
        
        # Detection predition
        bbox_outputs = self.bbox_embed(outputs['pooler_output']).sigmoid()
                
        return (classification_outputs, bbox_outputs)