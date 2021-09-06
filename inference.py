from torch.autograd import Variable
from make_vocab import *
import torch 
import re
from make_transformer_model import *
from optimizer import *
from training_setup import *
from torchtext.data.metrics import bleu_score

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

#Function to Load Checkpoint
def load_ckp(checkpoint_path, model, model_opt):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model_opt.optimizer.load_state_dict(checkpoint['optimizer'])
    return model, model_opt, checkpoint['epoch']
    
    
def main():
	
	data_path = "./data/multi30k/uncompressed_data"
	train_de, train_en, val_de, val_en, test_de, test_en = read_data_return_lists(data_path)
	
	dict_train_de_word_int, dict_train_de_int_word = make_vocab(train_de, language="de")
	dict_train_en_word_int, dict_train_en_int_word = make_vocab(train_en, language="en")
	
	checkpoint_dir = "/ssd_scratch/cvit/seshadri_c/temp/checkpoints/"
	checkpoint_path = checkpoint_dir + "checkpoint_latest.pt"
	
 	#Model made only with Train Vocab data	
	model = make_model(len(dict_train_de_word_int.keys()),len(dict_train_en_word_int.keys()), N=6)
	model_opt = get_std_opt(model)
	model.cuda()
	
	model, model_opt, epoch = load_ckp(checkpoint_path, model, model_opt)
	print("Loaded Successfully")
	
	target_list = []
	predicted_list = []
	
	for j in range(len(test_de)):
		sent = re.sub(r'[^\w\s]','',test_de[j]).split()
		temp= []
		for w in sent:
			try:
				temp.append(dict_train_de_word_int[w.upper()])
			except:
				temp.append(2)
		src = torch.LongTensor([temp])
		src = Variable(src)
		src_mask = (src != dict_train_de_word_int["<blank>"]).unsqueeze(-2)
		
		out = greedy_decode(model, src.cuda(), src_mask.cuda(), max_len=60, start_symbol=dict_train_de_word_int["<s>"])
		
		print("Translation:", end="\t")
		trans = ""
		for i in range(1, out.size(1)):
			sym = dict_train_en_int_word[int(out[0, i])]
			if sym == "</s>": 
				break
			trans += sym + " "
		
		target_list.append([test_en[j].upper().split()])
		predicted_list.append(trans.split())
		print("\n\n Pair : {}\n Target : {} \n Predicted : {}".format(j+1, target_list[-1], predicted_list[-1]))
		print("The BLEU Score : ",bleu_score(predicted_list, target_list)*100)

main()
