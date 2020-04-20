#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This is an example how to extend torch ranker agent and use it for your own purpose.
In this example, we will just use a simple bag of words model.
"""
from parlai.core.torch_ranker_agent import TorchRankerAgent
import torch
from torch import nn
from torch.nn import functional as F
from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.metrics import AverageMetric
from parlai.utils.misc import warn_once

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DualEncoder(nn.Module):
    """
    This constructs a simple bag of words model.
    It contains a encoder for encoding candidates and context.
    """

    def __init__(self, opt, dictionary):
        super().__init__()
        self.opt = opt
        hidden_dim = opt.get('hidden_dim', 512)
        embedding_size = opt.get('embedding_size', 128)
        num_layers = opt.get('num_layers', 1)

        self.dropout = opt.get('dropout', 0)
        self.dict = dictionary
        self.embeddings = nn.Embedding(len(dictionary), embedding_size)
        self.ctx_encoder = nn.GRU(
            input_size=embedding_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
        )
        self.cand_encoder = nn.GRU(
            input_size=embedding_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
        )
        #self.M = torch.zeros(hidden_dim, hidden_dim, requires_grad=True).cuda()
        #self.bias = torch.zeros(1, requires_grad=True).cuda()
        self.M = torch.empty(hidden_dim, hidden_dim).to(device)
        nn.init.uniform_(self.M, -0.01, 0.01)
        self.M.requires_grad = True

        self.bias = torch.empty(1).to(device)
        nn.init.uniform_(self.bias, -0.01, 0.01)
        self.bias.requires_grad = True
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cand_vecs, cand_encs=None):

        # compute the embedding for contexts
        contexts = batch['text_vec']
        contexts_emb = self.embeddings(contexts)
        ctx_output, ctx_hidden = self.ctx_encoder(contexts_emb)

        # reshape to match the number of candiates
        num_cands = cand_vecs.size(1)
        batch_size = ctx_hidden.size(1)
        hidden_size = ctx_hidden.size(-1)
        ctx_hidden = ctx_hidden.squeeze().view(-1, hidden_size).unsqueeze(1)
        ctx_hidden = ctx_hidden.expand(-1, num_cands, hidden_size)
        ctx_hidden = ctx_hidden.reshape(-1, hidden_size)
        ctx_hidden = F.dropout(ctx_hidden, p=self.dropout)

        # compute the embedding for candidates
        cand_vecs = cand_vecs.reshape(-1, cand_vecs.size(2))
        cands_emb = self.embeddings(cand_vecs)
        cand_output, cand_hidden = self.cand_encoder(cands_emb)
        cand_hidden = cand_hidden.squeeze()
        cand_hidden = torch.matmul(cand_hidden, self.M.t())
        cand_hidden = F.dropout(cand_hidden, p=self.dropout)

        # conver the score to sigmoid
        score = torch.sum(ctx_hidden * cand_hidden, 1).unsqueeze(1)
        score = score + self.bias
        score = self.sigmoid(score + self.bias)
        score = score.reshape(batch_size, num_cands)
        
        return score

class DualEncoderAgent(TorchRankerAgent):
    """
    Example subclass of TorchRankerAgent.
    This particular implementation is a simple bag-of-words model, which demonstrates
    the minimum implementation requirements to make a new ranking model.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add CLI args.
        """
        TorchRankerAgent.add_cmdline_args(argparser)
        arg_group = argparser.add_argument_group('DualEncoder Arguments')
        arg_group.add_argument('--hiddensize', type=int, default=512)
        arg_group.add_argument('--embedding_size', type=int, default=128)
        arg_group.add_argument('--num_layers', type=int, default=1)
        arg_group.add_argument('--dropout', type=float, default=0)
    
    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        This function takes in a Batch object as well as a Tensor of candidate vectors.
        It must return a list of scores corresponding to the likelihood that the
        candidate vector at that index is the proper response. If `cand_encs` is not
        None (when we cache the encoding of the candidate vectors), you may use these
        instead of calling self.model on `cand_vecs`.
        """
        scores = self.model.forward(batch, cand_vecs, cand_encs)
        print (batch['labels'][0])
        cands = batch['candidates'][0]
        print ("candidates: ")
        for cand in cands:
            print (cand)
        print (batch.keys())
        print (scores)
        print ('========================================')
        return scores

    def build_model(self):
        """
        This function is required to build the model and assign to the object
        `self.model`.
        """
        model = DualEncoder(self.opt, self.dict).to(device)
        return model

