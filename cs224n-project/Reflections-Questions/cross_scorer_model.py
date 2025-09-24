# Entirely copied from https://github.com/MichiganNLP/PAIR/blob/main/cross_scorer_model.py
import transformers
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch

from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers import BertForMaskedLM

import torch.nn.functional as F

# import spacy
import torch.nn as nn


class CrossScorerCrossEncoder(nn.Module):

    def __init__(self, transformer):

        super(CrossScorerCrossEncoder, self).__init__()

        self.cross_encoder = transformer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        # Binary Head
        self.l1 = torch.nn.Linear(768, 512)
        self.relu = torch.nn.ELU()
        self.l2 = torch.nn.Linear(512,1)

        self.encoder_type = "cross"    


    def score_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_attentions=False
        ):  


        output = self.cross_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pair_reps = output.last_hidden_state[:,0,:]
        score = self.l2(self.relu(self.l1(pair_reps)))
        
        if output_attentions and return_attentions:
            return score.sigmoid().squeeze(), output.attentions

        return score


    def cl_loss(self, pair_scores, labels):
        BSZ = pair_scores.size(0) 
        BSZ = int(BSZ/(4))

        pair_scores= list(pair_scores.tensor_split(BSZ, dim=0) )
        pair_scores = torch.stack(pair_scores)
        
        
        gap_1_loss_fct = nn.MarginRankingLoss(margin=0.5)
        gap_2_loss_fct = nn.MarginRankingLoss(margin=1.0)
              
        mq_scores = pair_scores[:,1] # 1
        lq_scores = pair_scores[:,2:-1] # 2


        hq_scores = pair_scores[:,0] 

        hq_mq_loss = gap_1_loss_fct(
                hq_scores.flatten(), 
                mq_scores.flatten(), 
                torch.ones(mq_scores.flatten().size()).to(self.device))
        mq_lq_loss = gap_1_loss_fct(
                mq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        hq_lq_loss = gap_2_loss_fct(
                hq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        
        mismatch_scores = pair_scores[:,-1]
        hq_mismatch_loss =  gap_2_loss_fct(
                        hq_scores.flatten(), 
                        mismatch_scores.flatten(), 
                        torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mq_mismatch_loss = gap_1_loss_fct(
                mq_scores.flatten(), 
                mismatch_scores.flatten(), 
                torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mismatch_loss = hq_mismatch_loss + mq_mismatch_loss 

        loss = hq_mq_loss + mq_lq_loss + hq_lq_loss  + mismatch_loss 
        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        random = False
        ):
 
        pair_scores = self.score_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).sigmoid().squeeze()


            
        cl_loss = self.cl_loss(pair_scores, labels)

        loss =   cl_loss 
        return SequenceClassifierOutput(
                loss=loss,
                logits=pair_scores,
                )
 