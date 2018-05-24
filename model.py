import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


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
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
        self.hidden = 0
    
    def forward(self, features, captions):
        self.hidden = (torch.zeros(self.num_layers, features.shape[0], self.hidden_size), torch.zeros(self.num_layers, features.shape[0], self.hidden_size))
        captions_transposed = torch.from_numpy(np.transpose(captions.cpu().numpy(), (1, 0))).cuda()
        embeds = self.word_embeddings(captions_transposed)
        print(embeds.shape)
        print("feature:", features.view(1, len(features), -1).shape)
        inputs = torch.cat([features.view(1, len(features), -1), embeds], 0)
        print("inputs:", inputs.shape)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        print("lstm_out:", lstm_out.shape)
        print("self.hidden:", self.hidden.shape)
        tag_space = self.hidden2tag(lstm_out)
        print("tag_space:", tag_space.shape)
        tag_scores = F.log_softmax(tag_space, dim=2)
        print("tag_scores:", tag_scores.shape)
        return torch.from_numpy(np.transpose(tag_scores.cpu().numpy(), (1, 0, 2))).cuda()

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass