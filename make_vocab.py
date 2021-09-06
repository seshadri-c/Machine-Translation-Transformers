
#Import Headers
from make_data_ready import *
import spacy

def read_data_return_lists(directory_path):
	
	train_de = [l.strip().upper() for l in open(os.path.join(directory_path,"train.de"), 'r', encoding='utf-8')]
	train_en = [l.strip().upper() for l in open(os.path.join(directory_path,"train.en"), 'r', encoding='utf-8')]
	val_de = [l.strip().upper() for l in open(os.path.join(directory_path,"val.de"), 'r', encoding='utf-8')]
	val_en = [l.strip().upper() for l in open(os.path.join(directory_path,"val.en"), 'r', encoding='utf-8')]
	test_de = [l.strip().upper() for l in open(os.path.join(directory_path,"test.de"), 'r', encoding='utf-8')]
	test_en = [l.strip().upper() for l in open(os.path.join(directory_path,"test.en"), 'r', encoding='utf-8')]
		
	return train_de, train_en, val_de, val_en, test_de, test_en
	
train_de, train_en, val_de, val_en, test_de, test_en = read_data_return_lists(uncompressed_data_path)

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
	
def make_vocab(sentences_list,language):
	
	#Beginning of Sentence
	BOS_WORD = '<s>'
	#End of Sentence
	EOS_WORD = '</s>'
	#Padding
	BLANK_WORD = "<blank>"
	
	dict_word_int = {}
	#Creating a Dictionary in the format {Token : Integer}
	i=0
	dict_word_int.update({BOS_WORD:i})
	i+=1
	dict_word_int.update({EOS_WORD:i})
	i+=1
	dict_word_int.update({BLANK_WORD:i})
	i+=1
	
	for s in sentences_list:
		if(language=="de"):
			token_list = tokenize_de(s)
		if(language=="en"):
			token_list = tokenize_en(s)
		for token in token_list:
			if token not in dict_word_int.keys():
				dict_word_int.update({token:i})
				i+=1			
	
	#Reversing the Dictionary in the format {Integer : Token}
	dict_int_word = {value : key for (key, value) in dict_word_int.items()}

	return dict_word_int, dict_int_word

def vocab_recheck(sentences, dict_word_int, language):
	i=0
	for s in sentences:
		if(language=="de"):
			tokens = tokenize_de(s)
		if(language=="en"):
			tokens = tokenize_en(s)	
		for t in tokens:
			if(t not in dict_word_int.keys()):	
				i+=1 
		return i
	
def read_data_return_lists(directory_path):
	
	train_de = [l.strip().upper() for l in open(os.path.join(directory_path,"train.de"), 'r', encoding='utf-8')]
	train_en = [l.strip().upper() for l in open(os.path.join(directory_path,"train.en"), 'r', encoding='utf-8')]
	val_de = [l.strip() for l in open(os.path.join(directory_path,"val.de"), 'r', encoding='utf-8')]
	val_en = [l.strip() for l in open(os.path.join(directory_path,"val.en"), 'r', encoding='utf-8')]
	test_de = [l.strip() for l in open(os.path.join(directory_path,"test.de"), 'r', encoding='utf-8')]
	test_en = [l.strip() for l in open(os.path.join(directory_path,"test.en"), 'r', encoding='utf-8')]
	
	train_de = train_de[:-1]
	train_en = train_en[:-1]
	val_de = val_de[:-1]
	val_en = val_en[:-1]
	
	return train_de, train_en, val_de, val_en, test_de, test_en
	
train_de, train_en, val_de, val_en, test_de, test_en = read_data_return_lists(uncompressed_data_path)


dict_train_de_word_int, dict_train_de_int_word = make_vocab(train_de, language="de")
dict_train_en_word_int, dict_train_en_int_word = make_vocab(train_en, language="en")

dict_test_de_word_int, dict_test_de_int_word = make_vocab(test_de, language="de")
dict_test_en_word_int, dict_test_en_int_word = make_vocab(test_en, language="en")

dict_val_de_word_int, dict_val_de_int_word = make_vocab(val_de, language="de")
dict_val_en_word_int, dict_val_en_int_word = make_vocab(val_en, language="en")

"""
if(vocab_recheck(train_de, dict_train_de_word_int, language="de")==0):
	print("Train De Successfull")

if(vocab_recheck(train_en, dict_train_en_word_int, language="en")==0):
	print("Train En Successfull")

if(vocab_recheck(test_de, dict_test_de_word_int, language="de")==0):
	print("Test De Successfull")

if(vocab_recheck(test_en, dict_test_en_word_int, language="en")==0):
	print("Test En Successfull")

if(vocab_recheck(val_de, dict_val_de_word_int, language="de")==0):
	print("Val De Successfull")

if(vocab_recheck(val_en, dict_val_en_word_int, language="en")==0):
	print("Val En Successfull")
	
"""
