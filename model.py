import torch
import torch.nn as nn
import torchvision.models as models

#----------------------------------------------------------------------------------------------------------
#                                                ENCODER
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
#----------------------------------------------------------------------------------------------------------
#                                              DECODER
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        #Implementing structure shown in the video from lesson 7.9
        #Embedding module. using embed_size as the dimension of the embedding_vector
        #and setting the number of embeddings to the size of the dictionary
        self.embedded=nn.Embedding(embedding_dim=embed_size,num_embeddings=vocab_size)
        #Long Shrt term memory cell
        self.lstm=nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,batch_first = True)
        #Implementing a dense layer (linear in pytorch). Use hidden_size as an input and vocab_size as an output
        self.linear=nn.Linear(in_features=hidden_size,out_features=vocab_size,bias=True)
        
    def forward(self, features, captions):
        #call the embedding layer from above and usethe captions as an input
        #make sure to remove the end token of the caption
        embedded=self.embedded(captions[:,:-1])
        #concatenate image features using torch.cat
        inputs = torch.cat((features.unsqueeze(1), embedded), 1)
        #call lstm, whicht will output the prediction and hidden state
        pred, hidden = self.lstm(inputs)
        #call linear with prediction as input
        linear_out=self.linear(pred)
        return linear_out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        #create empty list, where integer values are appended
        list=[]
        batch_size = inputs.shape[0]
        states = (torch.randn(1, 1, 512).to(inputs.device),torch.randn(1, 1, 512).to(inputs.device))
        for index in range(max_len):
            #call lstm to get prediction and hidden state
            pred, states = self.lstm(inputs,states)
            pred=pred.squeeze(1)
            #using the prediction from above as an input to the lnear layer
            linear_out=self.linear(pred)
            #linear_out.squeeze(0)
            #get the index if the highest value in linear_out
            max_out=linear_out.argmax(1)
            #append item to list
            list.append(max_out.item())
            if max_out.item()==1:
                break
            #define input the following iteration (use prediction)
            input=self.embedded(max_out).unsqueeze(1)
        return list
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    