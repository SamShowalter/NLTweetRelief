import torch
from torch import nn
from collections import namedtuple
import os

LSTMOutput = namedtuple('lstmoutput', ['logits'])
LSTMWithErrorOutput = namedtuple('lstmwitherroroutput', ['loss', 'logits'])

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, vocab_size, out_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings =  nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, out_size)
        self.criterion = nn.NLLLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # input ids: [batch_size, seq_len]
        embeds = self.word_embeddings(input_ids)
        # embeds: [batch_size, seq_len, embedding_dim]
        # lstm_layers input shape: [seq_len, batch_size, embedding_dim]
        lstm_out, state = self.lstm(embeds.transpose(0,1))
        # lstm_out: [batch_size, seq_len, hidden_dim]
        tag_space = self.hidden2tag(lstm_out.transpose(0,1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        if labels is not None:
            loss = self.criterion(tag_scores.view((-1, tag_scores.shape[-1])), labels.view((-1)))
            output = LSTMWithErrorOutput(loss, tag_scores)
            return output

        return LSTMOutput(tag_scores)

    def save_pretrained(self, folder):
        os.makedirs(folder)
        return torch.save(self.state_dict(), folder + "/model.pt")


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, vocab_size, out_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings =  nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, out_size)
        self.criterion = nn.NLLLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # input ids: [batch_size, seq_len]
        embeds = self.word_embeddings(input_ids)
        # embeds: [batch_size, seq_len, embedding_dim]
        # lstm_layers input shape: [seq_len, batch_size, embedding_dim]
        lstm_out, state = self.lstm(embeds.transpose(0,1))
        # lstm_out: [batch_size, seq_len, hidden_dim]
        # we take the last output of the lstm
        tag_space = self.hidden2tag(lstm_out.transpose(0,1)[-1])
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        if labels is not None:
            loss = self.criterion(tag_scores.view((-1, tag_scores.shape[-1])), labels.view((-1)))
            output = LSTMWithErrorOutput(loss, tag_scores)
            return output

        return LSTMOutput(tag_scores)

    def save_pretrained(self, folder):
        os.makedirs(folder)
        return torch.save(self.state_dict(), folder + "/model.pt")
