from torch.utils.data import Dataset, DataLoader
import os
import random
from make_vocab import *
import torch 
import numpy as np
from training_setup import *

class DataGenerator(Dataset):
	
	def __init__(self, path):
		self.files = self.get_files(path)
        

	def __len__(self):
		return len(self.files)
        

	def __getitem__(self,idx):

		f_de, f_en = self.files[idx]
		return f_de, f_en
			
	def get_files(self,path):

		data_de = [l.strip().upper() for l in open(path + ".de", 'r', encoding='utf-8')]
		data_en = [l.strip().upper() for l in open(path + ".en", 'r', encoding='utf-8')]
								
		#List to contain all the sentence pairs in tuples.
		files = []
		for i in range(len(data_de)-1):
			files.append((data_de[i],data_en[i]))
		
		return files[:10]		

#Tokenizing the batch of sentences and adding BOS_WORD and EOS_WORD 
def tokenize_and_add_BOS_EOS(sentence,language):

	#Beginning of Sentence
	BOS_WORD = '<s>'
	#End of Sentence
	EOS_WORD = '</s>'

	token_list = []
	token_list.append(BOS_WORD)
	if(language=="de"):
		token_list.extend(tokenize_de(str(sentence)))
	if(language=="en"):
		token_list.extend(tokenize_en(str(sentence)))
	token_list.append(EOS_WORD)
	
	return token_list
	
def padding(sent_batch,max_len):
	
	padded_sent_batch = []
	PAD_WORD = '<blank>'
	for s in sent_batch:
		[s.append(PAD_WORD) for i in range(len(s),max_len+2)]
		padded_sent_batch.append(s)
		
	return padded_sent_batch
	
def word_to_int(sent_batch,language):
	
	
	int_sent_batch = []
	
	if(language=="de"):
		dict_word_int = dict_train_de_word_int
		
	if(language=="en"):
		dict_word_int = dict_train_en_word_int
		
	for s in sent_batch:
		temp = []
		for t in s:
			try:
				temp.append(dict_word_int[t])
			except:
				temp.append(2)
				
		int_sent_batch.append(temp)
		
	return int_sent_batch
	
def collate_fn_customised(data):
	
	de_sent = []
	en_sent = []
	
	#Step 1 : Tokenization
	for d in data:
		de,en = d
		de_sent.append(tokenize_and_add_BOS_EOS(de,"de"))
		en_sent.append(tokenize_and_add_BOS_EOS(en,"en"))
	
	#Step 2 : Getting Maximum Length for a Batch
	max_de = max([len(s) for s in de_sent])
	max_en = max([len(s) for s in en_sent])
	
	#Step 3 : Padding the Sequences
	padded_de_sent = padding(de_sent, max_de)
	padded_en_sent = padding(en_sent, max_en)
	
	#Step 4 : Converting the Padded Batch to Integers
	int_de_sent = word_to_int(padded_de_sent,"de")
	int_en_sent = word_to_int(padded_en_sent,"en")
	
	#Step 5 : Get the Masks
	src = torch.tensor(np.array(int_de_sent))
	tgt = torch.tensor(np.array(int_en_sent))
	pad = 2
	src_mask, tgt_mask = make_std_mask(src, tgt, pad)
	

	return src, tgt, src_mask, tgt_mask
	
def load_data(data_path, batch_size=128, num_workers=2, shuffle=True):
    
	uncompressed_data_path  = "./data/multi30k/uncompressed_data"
	global dict_train_de_word_int, dict_train_de_int_word
	global dict_train_en_word_int, dict_train_en_int_word
	train_de, train_en, val_de, val_en, test_de, test_en = read_data_return_lists(uncompressed_data_path)
	dict_train_de_word_int, dict_train_de_int_word = make_vocab(train_de, language="de")
	dict_train_en_word_int, dict_train_en_int_word = make_vocab(train_en, language="en")

	dataset = DataGenerator(data_path)
	data_loader = DataLoader(dataset, collate_fn = collate_fn_customised, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

	return data_loader

#data_path = "./data/multi30k/uncompressed_data"
#load_data(data_path)
