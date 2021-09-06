from torch.autograd import Variable
from transformer.mask import *
from loss_backprop import *

def make_std_mask(src, tgt, pad):
    "Create a mask to hide padding and future words."
    #print("The shape of Source Data is : ",src.shape)
    src_mask = (src != pad).unsqueeze(-2)
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return src_mask, tgt_mask

class Batch:
    "Batch object."
    def __init__(self, src, trg, src_mask, trg_mask, ntokens):
        self.src = src
        self.trg = trg
        self.src_mask = src_mask
        self.trg_mask = trg_mask
        self.ntokens = ntokens
        
def train_epoch(train_iter, model, criterion, model_opt):
    "Standard Training and Logging Function"
    model.train()
    for i, batch in enumerate(train_iter):
        src, trg, src_mask, trg_mask = \
            batch.src, batch.trg, batch.src_mask, batch.trg_mask
            
        #print("The shapes are : ",src.shape,trg.shape)
        #print("The number of Tokens are : ",batch.ntokens)
        out = model.forward(src.cuda(), trg[:, :-1].cuda(), src_mask.cuda(), trg_mask[:, :-1, :-1].cuda())
        #print("The shape of out is : ",out.shape)
        #print("The shape of the target is : ",trg[:,1:].shape)
        loss = loss_backprop(model.generator, criterion, 
                             out, trg[:, 1:].cuda(), batch.ntokens) 
                        
        model_opt.step()
        model_opt.optimizer.zero_grad()
        if i % 10 == 1:
            print(i, loss, model_opt._rate)
            
def valid_epoch(valid_iter, model, criterion):
    "Standard validation function"
    model.eval()
    total = 0
    total_tokens = 0
    for batch in valid_iter:
        src, trg, src_mask, trg_mask = \
            batch.src, batch.trg, batch.src_mask, batch.trg_mask
        print("The shapes are (Validation): ",src.shape,trg.shape)
        print("The number of Tokens are (Validation): ",batch.ntokens)
        out = model.forward(src, trg[:, :-1], 
                            src_mask, trg_mask[:, :-1, :-1])
        total += loss_backprop(model.generator, criterion, out, trg[:, 1:], 
                             batch.ntokens, bp=False) * batch.ntokens
        total_tokens += batch.ntokens
    return total / total_tokens
