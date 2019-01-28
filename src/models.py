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


class Attention(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()

        self.W = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)


    def forward(self, word_embs, sentence_embs):
        """
        Args:
            word_embs: (batch_size, seq_len, embedding_dim)
            sentence_embs: (batch_size, num_sentence, embedding_dim)
        """

        batch_size, seq_len, embedding_dim = word_embs.size()
        num_sentence = sentence_embs.size(1)

        # Calculate h_1,t*W_a
        word_embs = word_embs.view(-1, embedding_dim)
        word_embs = self.W(word_embs)
        word_embs = word_embs.view(batch_size, seq_len, embedding_dim)

        # Caluclate score=(h_1,t*W_a)*s_j
        # att_scores shape: (batch_size, seq_len, num_sentence)
        att_scores = torch.bmm(word_embs, sentence_embs.transpose(1, 2))

        # Caluclate attention weights
        att_scores = att_scores.view(-1, num_sentence)
        att_weights = torch.nn.functional.softmax(att_scores, dim=-1)
        att_weights = att_weights.view(batch_size, seq_len, num_sentence)

        # shape: (batch_size, seq_len, embedding_dim)
        g = torch.bmm(att_weights, sentence_embs)

        # shape: (batch_size, seq_len, 2*embedding_dim)
        combination = torch.cat((word_embs, g), dim=-1)

        return combination, g