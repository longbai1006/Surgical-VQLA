'''
Description     : Gated Language-Vision Embedding module
Paper           : Surgical-VQLA: Transformer with Gated Vision-Language Embedding for 
                  Visual Question Localized-Answering in Robotic Surgery
Author          : Long Bai, Mobarakol Islam, Lalithkumar Seenivasan, Hongliang Ren
Lab             : Medical Mechatronics Lab, The Chinese University of Hong Kong
Acknowledgement : Code adopted from the official implementation of VisualBertModel from 
                  huggingface/transformers (https://github.com/huggingface/transformers.git)
                  and GMU (https://github.com/IsaacRodgz/ConcatBERT).
'''

from torch import nn
from transformers import VisualBertConfig
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializing a VisualBERT visualbert-vqa-coco-pre style configuration
config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
class VisualBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # For Visual Features
        # Token type and position embedding for image features
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)
        self.gated_linear = GatedMultimodalLayer(768*25, 768*25, 768*25)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
    ):

        input_shape = input_ids.size()
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        visual_embeds = self.visual_projection(visual_embeds)
        visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)
        visual_position_ids = torch.zeros(
            *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
        )
        visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)
        visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings

        embeddings = torch.flatten(embeddings, start_dim=1, end_dim=-1)
        visual_embeddings = torch.flatten(visual_embeddings, start_dim=1, end_dim=-1)        
        embeddings = self.gated_linear(embeddings, visual_embeddings)  
        embeddings = torch.reshape(embeddings, (-1, 25, 768))
      
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class GatedMultimodalLayer(nn.Module):
    def __init__(self, size_in1, size_in2, size_out):
        super().__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        # Weights hidden state modality 1
        weights_hidden1 = torch.Tensor(size_out, size_in1)
        self.weights_hidden1 = nn.Parameter(weights_hidden1)

        # Weights hidden state modality 2
        weights_hidden2 = torch.Tensor(size_out, size_in2)
        self.weights_hidden2 = nn.Parameter(weights_hidden2)

        # Weight for sigmoid
        weight_sigmoid = torch.Tensor(size_out*2)
        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        # initialize weights
        nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(torch.mm(x1, self.weights_hidden1.t()))
        h2 = self.tanh_f(torch.mm(x2, self.weights_hidden2.t()))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(torch.matmul(x, self.weight_sigmoid.t()))

        return z.view(z.size()[0],1)*h1 + (1-z).view(z.size()[0],1)*h2
