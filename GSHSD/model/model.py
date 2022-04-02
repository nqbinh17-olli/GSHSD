import os
import json
import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils import checkpoint
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model.layers.poolers import SqueezeAttentionPooling, CrossAttentionPooling
from model.layers.CNN import ConvBlock


class TransformerEncoder(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.
    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self, model_name_or_path: str,
                 classes_num: int,  
                 max_seq_length: Optional[int] = None,
                 dropout_rate:float = 0.2,
                 model_args: Dict = {}, 
                 cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, 
                 do_lower_case: bool = True,
                 conv_branch: bool = False,
                 cross_attention: bool = False,
                 residial: bool = False, 
                 tokenizer_name_or_path : str = None, 
                 checkpoint_batch_size : int = 1024):
        super(TransformerEncoder, self).__init__()
        #configs define
        self.config_keys = ['max_seq_length', 'do_lower_case', 'checkpoint_batch_size']
        self.do_lower_case = do_lower_case
        self.checkpoint_batch_size = checkpoint_batch_size
        self.residial = residial
        self.conv_branch = conv_branch
        self.cross_attention = cross_attention

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        tokenizer_path = tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path
        self.sent_encoder = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir, **tokenizer_args)
        self.drop_out_trans = nn.Dropout(dropout_rate)
        emb_size = self.get_word_embedding_dimension()

        if self.cross_attention:
            if self.conv_branch:
                self.CNN_model = ConvBlock(emb_size, emb_size)
            self.cross_pooler = CrossAttentionPooling(emb_size)
        
        self.classifier = nn.Linear(emb_size, classes_num)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

        #No max_seq_length set. Try to infer from model
        self.max_seq_length = self.__get_max_seq_length(max_seq_length) # Do nothing

        if tokenizer_name_or_path is not None:
            self.sent_encoder.config.tokenizer_class = self.tokenizer.__class__.__name__
    
    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.sent_encoder.__class__.__name__)
    
    def __get_max_seq_length(self, max_seq_length):
        if max_seq_length is None:
            max_seq_length = 32
            if self.__is_model_max_length_avaible():
                max_seq_length = min(self.sent_encoder.config.max_position_embeddings, self.tokenizer.model_max_length)
        return max_seq_length 
    
    def __is_model_max_length_avaible(self):
        tmp = hasattr(self.sent_encoder, "config") and hasattr(self.sent_encoder.config, "max_position_embeddings")
        return tmp and hasattr(self.tokenizer, "model_max_length")
    
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}
    
    def __partial_encode(self, *inputs):
        """ define function for checkpointing
        """
        encoder_outputs = self.sent_encoder.encoder(inputs[0], attention_mask=inputs[1], head_mask=inputs[2],) 
        return encoder_outputs.last_hidden_state # [bs, length, dim]
    
    def __embed_sentences_checkpointed(self, input_ids, attention_mask):
        device = input_ids.device
        input_shape = input_ids.size()
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        head_mask = [None] * self.sent_encoder.config.num_hidden_layers
        extended_attention_mask:torch.Tensor = self.sent_encoder.get_extended_attention_mask(attention_mask, input_shape, device)
        embedding_output = self.sent_encoder.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None)
        
        if input_shape[0] <= self.checkpoint_batch_size:
#           self.sent_encoder(input_ids, attention_mask=attention_mask).last_hidden_state # last_hidden_state [bs, length, dim]
            trans_outs = self.__partial_encode(embedding_output, extended_attention_mask, head_mask)
        else:
            trans_outs = []
            for b in range(math.ceil(input_shape[0] / self.checkpoint_batch_size)):
                b_embedding_output = embedding_output[b * self.checkpoint_batch_size : (b + 1) * self.checkpoint_batch_size]
                b_attention_mask = extended_attention_mask[b * self.checkpoint_batch_size : (b + 1) * self.checkpoint_batch_size]
                transformer_out_b = checkpoint.checkpoint(self.__partial_encode, b_embedding_output, b_attention_mask, head_mask)
                trans_outs.append(transformer_out_b)
            trans_outs = torch.cat(trans_outs, dim=0)
          
        transformer_out = self.drop_out_trans(trans_outs)
        return embedding_output, transformer_out # [bs, dim] X 2
        
    def get_word_embedding_dimension(self) -> int:
        return self.sent_encoder.config.hidden_size
        
    def save(self, output_path: str):
        self.sent_encoder.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        sbert_config_path = os.path.join(input_path, 'config.json')
        assert os.path.exists(sbert_config_path), f"Can not find \"config.json\" from {input_path}"
        
        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return TransformerEncoder(model_name_or_path=input_path, **config)
    
    def forward(self, features):
        embedding_output, transformer_out = self.__embed_sentences_checkpointed(features['input_ids'], features['attention_mask'])
        cls_ctx = transformer_out[:,0,:].squeeze()
        
        if self.cross_attention:
            if self.conv_branch:
                token_base_out = self.CNN_model(embedding_output) # cnn_out
            else:
                token_base_out = transformer_out
            
            global_local_emb = self.cross_pooler(cls_ctx, token_base_out)
            emb = global_local_emb + cls_ctx if self.residial else global_local_emb
        else:
            emb = cls_ctx
        
        return self.classifier(emb), emb
