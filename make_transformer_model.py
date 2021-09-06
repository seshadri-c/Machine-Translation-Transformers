import math, copy
from transformer.attention import *
from transformer.positionwise_feedforward import *
from transformer.positional_encoding import *
from transformer.encoder_decoder import *
from transformer.encoder import *
from transformer.decoder import *
from transformer.embeddings import *
from transformer.generator import *


def make_model(src_vocab, tgt_vocab, N=6,d_model=512, d_ff=2048, h=8, dropout=0.1):
   
	c = copy.deepcopy
	
	attn = MultiHeadedAttention(h, d_model, dropout)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)
	model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), 
		                     c(ff), dropout), N),
		nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
		nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
		Generator(d_model, tgt_vocab))
    
	# This was important from their code. 
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
		    nn.init.xavier_uniform_(p)
	return model
    
