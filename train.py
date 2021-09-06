from data_loader import *
from tqdm import tqdm
import numpy as np
import spacy
from make_transformer_model import *
from optimizer import *
from label_smoothing import *
from training_setup import *
from torchtext import data
from make_vocab import *
import random
import sys

use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")


#Function to Save Checkpoint
def save_ckp(checkpoint, checkpoint_path):
    torch.save(checkpoint, checkpoint_path)

#Function to Load Checkpoint
def load_ckp(checkpoint_path, model, model_opt):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model_opt.optimizer.load_state_dict(checkpoint['optimizer'])
    return model, model_opt, checkpoint['epoch']
    
    		
def train_epoch(epoch, train_loader, model, criterion, model_opt, dict_train_en_int_word):
	
	model.train()
	progress_bar = tqdm(enumerate(train_loader))
	total_loss = 0.0
	
	for step, (src, tgt, src_mask, tgt_mask) in progress_bar:
		out = model.forward(src.cuda(), tgt[:, :-1].cuda(), src_mask.cuda(), tgt_mask[:, :-1, :-1].cuda())
		ntokens = np.array(tgt[:,:-1]).shape[1]
		loss = loss_backprop(model.generator, criterion, out, tgt[:, 1:].cuda(), ntokens, dict_train_en_int_word)
		total_loss +=loss
		model_opt.step()
		model_opt.optimizer.zero_grad()
		progress_bar.set_description("Epoch : {} \t Training Loss : {}".format(epoch+1, total_loss / (step + 1))) 
		progress_bar.refresh()
		
	return total_loss/(step+1), model, model_opt			

def valid_epoch(epoch, valid_loader, model, criterion):
	
	model.eval()
	progress_bar = tqdm(enumerate(valid_loader))
	total_loss = 0.0
	total_tokens = 0
	for step, (src, tgt, src_mask, tgt_mask) in progress_bar:
		out = model.forward(src.cuda(), tgt[:, :-1].cuda(), src_mask.cuda(), tgt_mask[:, :-1, :-1].cuda())
		ntokens = np.array(tgt[:,:-1]).shape[1]
		loss = loss_backprop(model.generator, criterion, out, tgt[:, 1:].cuda(), ntokens, dict_train_en_int_word)
		total_loss +=loss
		progress_bar.set_description("Epoch : {} \t Validation Loss : {}".format(epoch+1, total_loss / (step + 1))) 
		progress_bar.refresh()	
		
	return total_loss/(step+1)
		
	

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    
    temp = torch.tensor([0], dtype=torch.long, device=device)
    
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(temp.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(temp.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(temp.data).fill_(next_word)], dim=1)
    return ys
    
def decode_target(cap_tensor):
	
	tgt = ""
	for t in np.array(cap_tensor.data.squeeze(0)):
		sym = dict_int_word[t]
		if sym == "</s>": 
			break
		if sym == "<s>":
			continue
		tgt += sym + " "
	return tgt
	
def test_epoch(epoch,valid_loader, model, criterion, dict_train_en_word_int, dict_train_en_int_word):
	
	progress_bar = tqdm(enumerate(valid_loader))
	model.eval()
	
	target_list = []
	predicted_list = []
	j=0
	
	original_stdout = sys.stdout
	with open("outputs/test_texts_"+str(epoch)+".txt", "a") as f:
		sys.stdout = f
		for step, (video_tensor, cap_tensor, src_mask, tgt_mask) in progress_bar:
			out = greedy_decode(model, video_tensor.to(device), src_mask.cuda(), max_len=60, start_symbol=dict_train_en_word_int["<s>"])
			
			trans = ""
			for i in range(1, out.size(1)):
				sym = dict_train_en_int_word[int(out[0, i])]
				if sym == "</s>": 
					break
				trans += sym + " "
				
			target_list.append([decode_target(cap_tensor).upper().split()])
			predicted_list.append(trans.upper().split())
			print("\n\n Pair : {}\n Target : {} \n Predicted : {}".format(j+1, target_list[-1], predicted_list[-1]))
			print("The BLEU Score : ",bleu_score(predicted_list, target_list)*100,"\n\n")
			j+=1
		sys.stdout = original_stdout
		
		
def	training_testing(train_loader, val_loader, model, criterion, model_opt, num_epochs, dict_train_de_word_int, dict_train_de_int_word, dict_train_en_word_int, dict_train_en_int_word):
	
	checkpoint_dir = "/ssd_scratch/cvit/seshadri_c/temp/checkpoints/"
	for epoch in range(num_epochs):
		
		checkpoint_path = checkpoint_dir + "checkpoint_latest.pt"
		checkpoint_duplicate_path = checkpoint_dir + "checkpoint_"+str(epoch+1)+".pt"
		print("\n\nTraining : ")
		print("Starting Epoch No : {}".format(epoch+1))
		total_train_loss, model, model_opt = train_epoch(epoch,train_loader, model, criterion, model_opt, dict_train_en_int_word)
		
		#Creating the Checkpoint
		checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(),'optimizer': model_opt.optimizer.state_dict()}
		#Saving the Checkpoint
		save_ckp(checkpoint, checkpoint_path)
		print("Saved Successfully")		
		save_ckp(checkpoint, checkpoint_duplicate_path)
		print("Saved Successfully")
		#Loading the Checkpoint
		model, model_opt, epoch = load_ckp(checkpoint_path, model, model_opt)
		print("Loaded Successfully")
		
		print("Epoch No {} completed. Total Training Loss : {}".format(epoch+1,total_train_loss))		
		total_val_loss = valid_epoch(epoch,val_loader, model, criterion)
		print("Epoch No {} completed. Total Validation Loss : {}".format(epoch+1,total_val_loss))
		
		#test_epoch(epoch,test_loader, model, criterion, dict_train_en_word_int, dict_train_en_int_word)
		
		
def main():
	
	data_path = "./data/multi30k/uncompressed_data"
	train_de, train_en, val_de, val_en, test_de, test_en = read_data_return_lists(data_path)
	
	dict_train_de_word_int, dict_train_de_int_word = make_vocab(train_de, language="de")
	dict_train_en_word_int, dict_train_en_int_word = make_vocab(train_en, language="en")
	
	#Model made only with Train Vocab data	
	model = make_model(len(dict_train_de_word_int.keys()),len(dict_train_en_word_int.keys()), N=6)
	model_opt = get_std_opt(model)
	model.cuda()
	
	#Input is the Target Vocab Size
	criterion = LabelSmoothing(size=len(dict_train_en_word_int.keys()), padding_idx=2, smoothing=0.1)
	criterion.cuda()
	
	train_loader = load_data(data_path+"/train", batch_size=128, num_workers=10, shuffle=True)
	val_loader = load_data(data_path+"/val", batch_size=128, num_workers=10, shuffle=True)
	#test_loader = load_data(data_path+"/test", batch_size=128, num_workers=10, shuffle=True)
	
	num_epochs = 20
	training_testing(train_loader, val_loader, model, criterion, model_opt, num_epochs, dict_train_de_word_int, dict_train_de_int_word, dict_train_en_word_int, dict_train_en_int_word)	
main()
