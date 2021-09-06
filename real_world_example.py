
from torchtext import data, datasets
import spacy
from make_transformer_model import *
from optimizer import *
from label_smoothing import *
from training_setup import *

#For English 
"""
Installation : python -m spacy download en_core_web_sm
Loading : english = spacy.load("en_core_web_sm")
"""

#For German
"""
Installation : python -m spacy download de_core_news_sm
Loading : german = spacy.load("de_core_news_sm")
"""

#Loading German 
spacy_de = spacy.load("de_core_news_sm")
#Loading English
spacy_en = spacy.load("en_core_web_sm")

#Tokenizer for German
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]
#Tokenizer for English
def tokenize_en(text):
	return [tok.text for tok in spacy_en.tokenizer(text)]
    
#Beginning of Sentence
BOS_WORD = '<s>'
#End of Sentence
EOS_WORD = '</s>'
#Padding
BLANK_WORD = "<blank>"
 
SRC = data.Field(tokenize=tokenize_de, init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD)  
MAX_LEN = 100
train, val, test = datasets.Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TGT), filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)                                 

MIN_FREQ = 1
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
print("The integers are : ",SRC.vocab.stoi[BOS_WORD],SRC.vocab.stoi[EOS_WORD],SRC.vocab.stoi[BLANK_WORD]) 
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
# Batching matters quite a bit. 
# This is temporary code for dynamic batching based on number of tokens.
# This code should all go away once things get merged in this library.

BATCH_SIZE = 30
global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    #print ("The max Size is : ",max(src_elements, tgt_elements))
    return max(src_elements, tgt_elements)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
            
			    #EXPLAINING THE sort_key
			    #sort_key: A key to use for sorting examples in order to batch together
		        #examples with similar lengths and minimize padding. The sort_key
		        #provided to the Iterator constructor overrides the sort_key
		        #attribute of the Dataset, or defers to it if None.
		        
		        #EXPLAINING THE batch_size_fn19seshadri97
		        
				#batch_size_fn: Function of three arguments (new example to add, current
				#count of examples in the batch, and current effective batch size)
				#that returns the new effective batch size resulting from adding
				#that example to a batch. This is useful for dynamic batching, where
				#this function would add to the current effective batch size the
				#number of tokens in the new example.
		       
            	for p in data.batch(d, self.batch_size * 100):
            		p_batch = data.batch(sorted(p, key=self.sort_key),self.batch_size, self.batch_size_fn)
            		for b in random_shuffler(list(p_batch)):
            			yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    #print("The batch is : ",batch)
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    src_mask, trg_mask = make_std_mask(src, trg, pad_idx)
    return Batch(src, trg, src_mask, trg_mask, (trg[1:] != pad_idx).data.sum())

train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True)
valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)
# Create the model an load it onto our GPU.
pad_idx = TGT.vocab.stoi["<blank>"]
model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
model_opt = get_std_opt(model)
model.cuda()
None
criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.cuda()
for epoch in range(5):
	train_epoch((rebatch(pad_idx, b) for b in train_iter), model, criterion, model_opt)
	print(valid_epoch((rebatch(pad_idx, b) for b in valid_iter), model, criterion))
