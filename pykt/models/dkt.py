import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout

class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c  # 知识组件的数量（概念concepts 的数量）
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        

    def forward(self, q, r):
        # print(f"q.shape is {q.shape}")
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        # print(f"xemb.shape is {xemb.shape}")
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y
    

if __name__ == "__main__":
    from torch.nn.functional import one_hot
    
    batch_size = 256
    n = 200
    num_c = 100
    emb_size = 200

    q = torch.tensor(np.random.randint(0,num_c,size=(batch_size, n-1)))
    cshft = torch.tensor(np.random.randint(0,num_c,size=(batch_size, n-1)))
    r = torch.tensor(np.random.randint(0,2,size=(batch_size, n-1)))

    model = DKT(num_c, emb_size)
    y = model(q, r)
    print(y.shape)
    y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
    print(y.shape)