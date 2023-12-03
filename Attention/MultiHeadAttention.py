import torch
import torch.nn as nn
import math
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)

    def forward(self,q,k,v,mask=None):
        q_k_t = torch.bmm(q,k.transpose(1,2))
        u = q_k_t/math.sqrt(k.size(-1))
        if mask is not None:
            u = u.masked_fill(mask, -np.inf)
        attn = self.softmax(u)
        output = torch.bmm(attn,v)
        return output

class MultiHeadAttention(nn.Module):
    """
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)
        
    """
    def __init__(self,n_head,infeatures,dv):
        super().__init__()
        self.n_head = n_head
        self.features = infeatures
        self.d_v = dv //n_head
        self.d_k = infeatures // n_head
        self.linears = nn.Linear(infeatures,infeatures)
        self.attention = ScaledDotProductAttention()

    def forward(self,q,k,v,mask=None):
        """
        多头注意力机制的前向传播。

        参数:
        - q (torch.Tensor): 查询张量。
        - k (torch.Tensor): 键张量。
        - v (torch.Tensor): 值张量。
        - mask (torch.Tensor, 可选): 用于应用于注意力分数的掩码。默认为None。

        返回:
        - y (torch.Tensor): 多头注意力机制后的结果张量。
        """
        batch,q_n,dq = q.size()
        batch,k_n,dk = k.size()
        batch,v_n,dv = v.size()
        q = self.linears(q)
        k = self.linears(k)
        q = q.view(batch,q_n,self.n_head,self.d_k).permute(0,2,1,3).contiguous().view(batch*self.n_head,-1,self.d_k)
        k = k.view(batch,k_n,self.n_head,self.d_k).permute(0,2,1,3).contiguous().view(batch*self.n_head,-1,self.d_k)
        v = v.view(batch,v_n,self.n_head,self.d_v).permute(0,2,1,3).contiguous().view(batch*self.n_head,-1,self.d_v)
        #print(q.size())
        #print(v.size())
        y = self.attention(q,k,v,mask)
        y = y.view(batch,self.n_head,-1,self.d_v).permute(0,2,1,3).contiguous().view(batch,-1,self.n_head*self.d_v)
        return y

if __name__ == "__main__":
    batch = 3
    infeatures = 512
    seqlen = 100
    dv =infeatures//2
    q = torch.randn(batch,seqlen,infeatures)
    k = torch.randn(batch,seqlen,infeatures)
    v = torch.randn(batch,seqlen,dv)
    sinattention = ScaledDotProductAttention()
    mulattention = MultiHeadAttention(n_head=8,infeatures=infeatures,dv=dv)

    output2 = mulattention(q,k,v)
    output1 = sinattention(q,k,v)
    print(output1.size())
    print(output2.size()) 