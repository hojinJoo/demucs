import torch
from torch.nn.functional import cross_entropy

def l2_normalize(x) :
    norm = torch.linalg.norm(x,dim=-1,keepdim=True)
    return x / norm.clamp(min=1e-10)

def cos_pairwise(x,y=None) :
    x = l2_normalize(x)
    y = l2_normalize(y) if y is not None else x
    N = x.shape[:-1]
    M = y.shape[:-1]
    x = x.flatten(end_dim=-2)
    y = y.flatten(end_dim=-2)
    cos = torch.einsum("nc,mc->nm", x,y)

    return cos.reshape(N+M)




def ctr_loss(data) :
    B,N,D = data.shape
    cos = cos_pairwise(data)
    positives = get_positive_pair(data.size())
    
    negative_inter = get_inter_negative_pair(data.size(),2,4)
    negative_intra = get_intra_negative_pair(data.size(),2,4)
    
    negatives = torch.cat([negative_inter,negative_intra],dim=0)
    
    cos_pos = cos[positives[:,0],positives[:,1],positives[:,2],positives[:,3]]
    cos_neg = cos[negatives[:,0],negatives[:,1],negatives[:,2],negatives[:,3]]
    
    
    
    
    
def get_positive_pair(size) :
    slot_idx = torch.arange(size[1])
    batch_idx = get_batch_idx(size,4)

    positive = torch.stack([batch_idx[:,0],slot_idx,batch_idx[:,1],slot_idx],dim=-1)
    return positive


def get_inter_negative_pair(size,slot_count,batch_count) :
    slot_idx = torch.stack( [ torch.stack([torch.randperm(size[1])[:2] for _ in range(slot_count)],dim=0)  for _ in range(batch_count)] ,dim=0)
    batch_idx = torch.stack([torch.randperm(size[0])[:1] for _ in range(batch_count)],dim=0)
    
    inter_negative_pair = []
    for b in range(batch_count) :
        for s in range(slot_count) :
            inter_negative_pair.append(torch.stack([batch_idx[b,0],slot_idx[b,s,0],batch_idx[b,0],slot_idx[b,s,1]],dim=0))
    inter_negative_pair = torch.stack(inter_negative_pair,dim=0)
    
    return inter_negative_pair
    
def get_intra_negative_pair(size,slot_count,batch_count) :
    slot_idx = torch.stack( [ torch.stack([torch.randperm(size[1])[:2] for _ in range(slot_count)],dim=0)  for _ in range(batch_count)] ,dim=0)
    
    batch_idx = get_batch_idx(size,batch_count)
    
    
    inter_negative = []
    for b in range(batch_count) :
        for s in range(slot_count) :
    
            inter_negative.append(torch.stack([batch_idx[b,0],slot_idx[b,s,0],batch_idx[b,1],slot_idx[b,s,1]],dim=0))
    intra_negative = torch.stack(inter_negative,dim=0)
    
    return intra_negative
    
def get_batch_idx(size,counts) :
    b,n_s,d = size 
    batch_idx = torch.stack([torch.randperm(b)[:2] for _ in range(counts) ],dim=0)
    return batch_idx
    
if __name__ == '__main__' :
    batch_size =16
    n_slot = 4
    data = torch.rand((batch_size,n_slot,768))
    ctr_loss(data)