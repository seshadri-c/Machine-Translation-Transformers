from label_smoothing import *
from make_transformer_model import *
from optimizer import *
from training_setup import *

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    #Iterating for Number of Batches.
    for i in range(nbatches):
    
    	#Generating data with shape = (batch x 10) and the data range is in [1,V).
    	#Then converting the numpy matrix to torch.tensor()
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        print("The shape of the data is : ",data.shape)
        
        src = Variable(data, requires_grad=False)
        print("The shape of the Source is after converting it to Variable : ",src.shape)
        tgt = Variable(data, requires_grad=False)
        print("The shape of the Target is after converting it to Variable : ",tgt.shape)
        
        #GENERATING THE MASKS FOR THE DATA
        src_mask, tgt_mask = make_std_mask(src, tgt, 0)
        
        #Source Mask is Full of True
                
        #Shape of Source Mask : data.shape[0] x 1 x data.shape[1]
        print("The shape of the Source Mask is : ",src_mask.shape)
        "The shape of the Source Mask is :  torch.Size([5, 1, 10])"
        
        #Example : (Source Mask when Data is of shape : 5 x 10)
        #print("The Source Mask is : \n",src_mask)
        """
		The Source Mask is : 
		 tensor([[[True, True, True, True, True, True, True, True, True, True]],

				[[True, True, True, True, True, True, True, True, True, True]],

				[[True, True, True, True, True, True, True, True, True, True]],

				[[True, True, True, True, True, True, True, True, True, True]],

				[[True, True, True, True, True, True, True, True, True, True]]])
        """
        
        #Shape of Target Mask : data.shape[0] x data.shape[1] x data.shape[1]
        print("The shape of the Target Mask is : ",tgt_mask.shape)
        "The shape of the Target Mask is :  torch.Size([5, 10, 10])"
        
        #Example : (Target Mask when Data is of shape : 5 x 10)
        #print("The Target Mask is : \n",tgt_mask)
        """
        The Target Mask is : 
		 tensor([[[ True, False, False, False, False, False, False, False, False, False],
				 [ True,  True, False, False, False, False, False, False, False, False],
				 [ True,  True,  True, False, False, False, False, False, False, False],
				 [ True,  True,  True,  True, False, False, False, False, False, False],
				 [ True,  True,  True,  True,  True, False, False, False, False, False],
				 [ True,  True,  True,  True,  True,  True, False, False, False, False],
				 [ True,  True,  True,  True,  True,  True,  True, False, False, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True,  True, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]],

				[[ True, False, False, False, False, False, False, False, False, False],
				 [ True,  True, False, False, False, False, False, False, False, False],
				 [ True,  True,  True, False, False, False, False, False, False, False],
				 [ True,  True,  True,  True, False, False, False, False, False, False],
				 [ True,  True,  True,  True,  True, False, False, False, False, False],
				 [ True,  True,  True,  True,  True,  True, False, False, False, False],
				 [ True,  True,  True,  True,  True,  True,  True, False, False, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True,  True, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]],

				[[ True, False, False, False, False, False, False, False, False, False],
				 [ True,  True, False, False, False, False, False, False, False, False],
				 [ True,  True,  True, False, False, False, False, False, False, False],
				 [ True,  True,  True,  True, False, False, False, False, False, False],
				 [ True,  True,  True,  True,  True, False, False, False, False, False],
				 [ True,  True,  True,  True,  True,  True, False, False, False, False],
				 [ True,  True,  True,  True,  True,  True,  True, False, False, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True,  True, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]],

				[[ True, False, False, False, False, False, False, False, False, False],
				 [ True,  True, False, False, False, False, False, False, False, False],
				 [ True,  True,  True, False, False, False, False, False, False, False],
				 [ True,  True,  True,  True, False, False, False, False, False, False],
				 [ True,  True,  True,  True,  True, False, False, False, False, False],
				 [ True,  True,  True,  True,  True,  True, False, False, False, False],
				 [ True,  True,  True,  True,  True,  True,  True, False, False, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True,  True, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]],

				[[ True, False, False, False, False, False, False, False, False, False],
				 [ True,  True, False, False, False, False, False, False, False, False],
				 [ True,  True,  True, False, False, False, False, False, False, False],
				 [ True,  True,  True,  True, False, False, False, False, False, False],
				 [ True,  True,  True,  True,  True, False, False, False, False, False],
				 [ True,  True,  True,  True,  True,  True, False, False, False, False],
				 [ True,  True,  True,  True,  True,  True,  True, False, False, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True, False, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True,  True, False],
				 [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True]]])
        """
        
        yield Batch(src, tgt, src_mask, tgt_mask, (tgt[1:] != 0).data.sum())

# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

#Number of layers in the Transformer
N = 2

#Generating the Model
model = make_model(V, V, N)

#Getting the Optimizer for the Model
model_opt = get_std_opt(model)

#Setting the number of Epochs
num_epoch = 1

#Iterating on the number of Epochs
for epoch in range(num_epoch):

	#Calling the Training Function
    train_epoch(data_gen(V, 5, 20), model, criterion, model_opt)
    #print(valid_epoch(data_gen(V, 30, 5), model, criterion))
