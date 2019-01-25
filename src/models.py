import torch

class BiLM(torch.nn.Module):
    def __init__(self, embedding_dim, lstm_units, vocab_size):
        super(BiLM, self).__init__()

        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bi_lstm = torch.nn.LSTM(embedding_dim, lstm_units, num_layers=1, bidirectional=True)
        self.linear = torch.nn.Linear(lstm_units, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        output, (h, c) = self.bi_lstm(emb)
        forward_output, backword_output = output[:, :, :self.lstm_units], output[:, :, self.lstm_units:]
        
        # shape: (batch_size * timesteps, lstm_units)
        forward_mask, backward_mask = self._get_mask(x)
        forward_output = forward_output[forward_mask]
        backword_output = backword_output[backward_mask]
        
        # Log-softmax
        forward_output = self._calc_proba(forward_output)
        backword_output = self._calc_proba(backword_output)
        
        return forward_output, backword_output, c

    def _calc_proba(self, embeddings):
        return torch.nn.functional.log_softmax(self.linear(embeddings), dim=-1)
    
    def _get_mask(self, inputs):
        forward_mask = torch.cat((inputs[1:], inputs[:1])) > 2
        backward_mask = torch.cat((inputs[-1:], inputs[:-1])) > 2
        
        return forward_mask, backward_mask