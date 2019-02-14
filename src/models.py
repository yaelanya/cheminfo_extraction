import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLM(nn.Module):
    def __init__(self, embedding_dim, lstm_units, vocab_size):
        super(BiLM, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bi_lstm = nn.LSTM(embedding_dim, lstm_units, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(lstm_units, vocab_size)

    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, seq_len)
        """
        embs = self.embedding(inputs)
        embs = embs.transpose(1, 0) # (seq_len, batch_size, embedding_dim)
        output, (h, c) = self.bi_lstm(embs)
        output = output.transpose(1, 0) # (batch_size, seq_len, 2*lstm_units)
        forward_output, backword_output = output[:, :, :self.lstm_units], output[:, :, self.lstm_units:]
        
        # shape: (batch_size * timesteps, lstm_units)
        forward_mask, backward_mask = self._get_mask(inputs)
        forward_output = forward_output.masked_select(forward_mask.unsqueeze(-1)).view(-1, self.lstm_units)
        backword_output = backword_output.masked_select(backward_mask.unsqueeze(-1)).view(-1, self.lstm_units)

        # shape: (batch_size * timesteps, vocab_size)
        forward_output = nn.functional.log_softmax(self.linear(forward_output), dim=-1)
        backword_output = nn.functional.log_softmax(self.linear(backword_output), dim=-1)
        
        loss = self._softmax_loss(forward_output, backword_output, inputs)

        return loss, h, c

    def _softmax_loss(self, forward_output, backward_output, targets):
        # target shape: (batch_size * timesteps,)
        if targets.dim() > 1:
            targets = targets.view(-1)

        # except PAD, BOS and EOS
        targets = targets[targets > 2]
        num_targets = torch.sum(targets > 2)

        forward_loss = torch.nn.functional.nll_loss(forward_output, targets, reduction='sum')
        backward_loss = torch.nn.functional.nll_loss(backward_output, targets, reduction='sum')

        if num_targets > 0:
            average_loss = 0.5 * (forward_loss + backward_loss) / num_targets
        else:
            average_loss = torch.tensor(0.0).to(self.device)

        return average_loss
    
    def _get_mask(self, inputs):
        forward_mask = torch.cat((inputs[:, 1:], inputs[:, :1]), dim=1) > 2
        backward_mask = torch.cat((inputs[:, -1:], inputs[:, :-1]), dim=1) > 2
        
        return forward_mask, backward_mask


class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()

        self.W = nn.Linear(embedding_dim, embedding_dim, bias=False)


    def forward(self, word_embs, sentence_embs):
        """
        Args:
            word_embs: (batch_size, seq_len, embedding_dim)
            sentence_embs: (batch_size, num_sentence, embedding_dim)
        """

        batch_size, seq_len, embedding_dim = word_embs.size()
        num_sentence = sentence_embs.size(1)

        # Calculate h_1,t*W_a
        word_embs = word_embs.contiguous()
        word_embs = word_embs.view(-1, embedding_dim)
        word_embs = self.W(word_embs)
        word_embs = word_embs.view(batch_size, seq_len, embedding_dim)

        # Caluclate score=(h_1,t*W_a)*s_j
        # att_scores shape: (batch_size, seq_len, num_sentence)
        att_scores = torch.bmm(word_embs, sentence_embs.transpose(1, 2))

        # Caluclate attention weights
        att_scores = att_scores.view(-1, num_sentence)
        att_weights = nn.functional.softmax(att_scores, dim=-1)
        att_weights = att_weights.view(batch_size, seq_len, num_sentence)

        # shape: (batch_size, seq_len, embedding_dim)
        g = torch.bmm(att_weights, sentence_embs)

        # shape: (batch_size, seq_len, 2*embedding_dim)
        combination = torch.cat((word_embs, g), dim=-1)

        return combination, g

class CRF(nn.Module):
    def __init__(self, tag_to_ix):
        super(CRF, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.PAD_TAG = "<PAD>"
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"

        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[self.START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[self.STOP_TAG]] = -10000

    def forward(self, feats):
        """
        Args:
            feats: (seq_len, tagset_size)
        Outputs:
            output: (seq_len)
                - output is predicted tag sequence
        """
        return self._viterbi_decode(feats)

    def forward_alg(self, feats, tags):
        """
        Args:
            feats: (batch_size, seq_len, tagset_size)
            tags: (batch_size, seq_len)
        Outputs:
            forward_score: (batch_size)
        """
        batch_size = feats.size(0)

        alpha = torch.full((batch_size, self.tagset_size), -10000.).to(self.device.type)
        alpha[:, self.tag_to_ix[self.START_TAG]] = 0.

        feats = feats.transpose(1, 0) # (seq_len, batch_size, tag_size)
        tags = tags.transpose(1, 0) # (seq_lenm batch_size)
        for feat, tag in zip(feats, tags):
            emit_score = feat.unsqueeze(-1).expand(batch_size, *self.transitions.size())
            alpha_t = alpha.unsqueeze(1).expand(batch_size, *self.transitions.size())
            trains_t = self.transitions.expand(batch_size, *self.transitions.size())
            next_tag_var = alpha_t + trains_t + emit_score
            next_alpha = self._log_sum_exp(next_tag_var, dim=-1)

            mask = (tag != self.tag_to_ix[self.PAD_TAG]).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * next_alpha + (1 - mask) * alpha

        terminal_var = alpha + self.transitions[self.tag_to_ix[self.STOP_TAG]].expand(*alpha.size())

        return self._log_sum_exp(terminal_var, dim=-1)

    def transition_score(self, tags):
        """
        Args:
            tags: (batch_size, seq_len)
        Outputs:
            transition_score: (batch_size)
        """
        batch_size, seq_len = tags.size()
    
        tags_t = torch.full((batch_size, seq_len + 2), self.tag_to_ix[self.STOP_TAG], dtype=torch.long).to(self.device.type)
        tags_t[:, 0] = self.tag_to_ix[self.START_TAG]
        tags_t[:, 1:-1] = tags

        mask = (tags_t[:, :-1] != self.tag_to_ix[self.PAD_TAG]).float()

        tags_t[tags_t == self.tag_to_ix[self.PAD_TAG]] = self.tag_to_ix[self.STOP_TAG]
        
        next_tags = tags_t[:, 1:]
        next_tags = next_tags.unsqueeze(-1).expand(*next_tags.size(), self.transitions.size(0))
        trans_t = self.transitions.expand(batch_size, *self.transitions.size())
        trans_row = torch.gather(trans_t, 1, next_tags)
        
        prev_tags = tags_t[:, :-1]
        prev_tags = prev_tags.unsqueeze(-1)
        score = torch.gather(trans_row, 2, prev_tags).squeeze(-1)
        score = score * mask

        return score.squeeze(-1).sum(-1)

    def _viterbi_decode(self, feats):
        """
        Args:
            inputs: (seq_len, tagset_size)
        Outputs:
            output: (seq_len)
                - output is predicted tag sequence
        """
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device.type)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self._argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = self._argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _log_sum_exp(self, vec, dim=0):
        _max, _ = torch.max(vec, dim)
        max_exp = _max.unsqueeze(-1).expand_as(vec)
        return _max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))

    def _argmax(self, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, -1)
        return idx.item()

class BiLSTM_CRF(nn.Module):
    def __init__(self
                 , tag_to_ix
                 , word_vocab_size
                 , word_embedding_dim
                 , word_lstm_units
                 , char_vocab_size
                 , char_embedding_dim
                 , char_lstm_units
                 , fc_dim
                 , dropout=0.5):
        super(BiLSTM_CRF, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.PAD_TAG = "<PAD>"

        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_lstm_units = word_lstm_units
        self.char_vocab_size = char_vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.char_lstm_units = char_lstm_units
        self.fc_dim = fc_dim
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.char_embeds = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_units, num_layers=1, bidirectional=True)
        self.word_embeds = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.word_lstm = nn.LSTM(word_embedding_dim + 2*char_lstm_units, word_lstm_units, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(2 * word_lstm_units, fc_dim)
        self.hidden2tag = nn.Linear(fc_dim, self.tagset_size)
        self.crf = CRF(tag_to_ix)

    def forward(self, word_inputs, char_inputs):
        """
        Args:
            word_inputs: (batch_size, seq_len)
            char_inputs: (batch_size, seq_len, word_len)
        Outputs:
            output: (batch_size, seq_len)
                - output is predicted tag sequence
        """
        lstm_feats = self._get_lstm_features(word_inputs, char_inputs)
        tag_seq_batch = []
        for feats in lstm_feats:
            _, tag_seq = self.crf(feats)
            tag_seq_batch.append(tag_seq)

        return torch.tensor(tag_seq_batch).to(self.device.type)  # (batch_size, seq_len)

    def _get_lstm_features(self, word_inputs, char_inputs):
        """
        Args:
            word_inputs: (batch_size, seq_len)
            char_inputs: (batch_size, seq_len, word_len)
        Outputs:
            output: (batch_size, seq_len, tagset_size)
        """
        batch_size = word_inputs.size(0)

        lens = (word_inputs > 0).sum(-1)
        sorted_lens, sorted_indices = lens.sort(descending=True)
        _, origin_indices = sorted_indices.sort(0)
        
        # Character embeddings
        char_embs = self._get_char_feats(char_inputs) # (batch_size, seq_len, 2*char_embedding_dim)
        char_embs = char_embs.transpose(1, 0) # (seq_len, batch_size, 2*char_embedding_dim)
        
        # Word embeddings
        word_embs = self.word_embeds(word_inputs) # (batch_size, seq_len, word_embedding_dim)
        word_embs = word_embs.transpose(1, 0) # (seq_len, batch_size, word_embedding_dim)

        embeds = torch.cat((word_embs, char_embs), dim=-1)
        
        embeds = self.dropout(embeds)

        embeds = embeds[sorted_indices]
        packed = pack_padded_sequence(embeds, sorted_lens)
        packed_lstm_out, _ = self.word_lstm(packed)
        lstm_out = pad_packed_sequence(packed_lstm_out) # (seq_len, batch_size, 2*word_lstm_units)
        lstm_out = lstm_out[origin_indices]

        lstm_out = lstm_out.transpose(1, 0)
        lstm_out = lstm_out.contiguous()
        lstm_out = lstm_out.view(-1, 2*self.word_lstm_units) # (batch_size*seq_len, 2*lstm_units)
        
        lstm_feats = self.fc(lstm_out)
        lstm_feats = self.hidden2tag(lstm_feats) # (batch_size*seq_len, tagset_size)

        return lstm_feats.view(batch_size, -1, self.tagset_size) # (batch_size, seq_len, tagset_size)

    def _get_char_feats(self, char_inputs):
        """
        Args:
            char_inputs: (batch_size, seq_len, word_len)
        Outputs:
            char_embeddings: (batch_size, seq_len, 2*char_embedding_dim)
        """
        batch_size, seq_len, word_len = char_inputs.size()

        char_inputs = char_inputs.view(-1, word_len) # (batch_size*seq_len, word_len)

        lens = (char_inputs > 0).sum(-1)
        sorted_lens, sorted_indices = lens.sort(descending=True)
        _, origin_indices = sorted_indices.sort(0)

        char_embs = self.char_embeds(char_inputs[sorted_indices])
        packed = pack_padded_sequence(char_embs[sorted_lens > 0], sorted_lens[sorted_lens > 0], batch_first=True)
        _, (h_n, _) = self.char_lstm(packed)

        forward_last_state, backward_last_state = h_n[0], h_n[1]
        char_feats = torch.cat((forward_last_state, backward_last_state), dim=-1)

        count_pad_word = (sorted_lens == 0).sum()
        pad_word = torch.zeros((count_pad_word, 2*self.char_lstm_units)).to(self.device)
        char_feats = torch.cat((char_feats, pad_word))
        char_feats = char_feats[origin_indices]

        return char_feats.view(batch_size, seq_len, -1)

    def _score_sentence(self, feats, tags):
        transition_score = self.crf.transition_score(tags)
        lstm_score = self._lstm_score(feats, tags)
        return transition_score + lstm_score
    
    def _lstm_score(self, feats, tags):
        mask = (tags != self.tag_to_ix[self.PAD_TAG]).float()
        tags = tags.unsqueeze(-1)
        score = torch.gather(feats, 2, tags).squeeze(-1)
        score = mask * score

        return score.sum(-1)

    def neg_log_likelihood(self, word_inputs, char_inputs, targets):
        """
        Args:
            inputs: (batch_size, seq_len)
            targets: (batch_size, seq_len)
            ignore_index: int
        Outputs:
            loss: Negative log likelihood loss
        """
        feats = self._get_lstm_features(word_inputs, char_inputs)
        losses = self.crf.forward_alg(feats, targets) - self._score_sentence(feats, targets)

        return losses.mean()

class Att_BiLSTM_CRF(nn.Module):
    """
    reference: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    """
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, lstm1_units, lstm2_units, dropout=0.5):
        super(Att_BiLSTM_CRF, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.PAD_TAG = "<PAD>"

        self.embedding_dim = embedding_dim
        self.lstm1_units = lstm1_units
        self.lstm2_units = lstm2_units
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm_1 = nn.LSTM(embedding_dim, lstm1_units, num_layers=1, bidirectional=True)
        self.att = Attention(2*lstm1_units)
        self.lstm_2 = nn.LSTM(2*2*lstm1_units, lstm2_units, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(2*lstm2_units, self.tagset_size)
        self.crf = CRF(tag_to_ix)

    def forward(self, inputs, sent_embs):
        """
        Args:
            inputs: (batch_size, seq_len)
            sent_feats: (batch_size, num_sentence, embedding_dim)
        Outputs:
            output: (batch_size, seq_len)
                - output is predicted tag sequence
        """
        lstm_feats = self._get_lstm_features(inputs, sent_embs)
        tag_seq_batch = []
        for feats in lstm_feats:
            _, tag_seq = self.crf(feats)
            tag_seq_batch.append(tag_seq)

        return torch.tensor(tag_seq_batch).to(self.device.type)  # (batch_size, seq_len)

    def _get_lstm_features(self, inputs, sent_embs):
        """
        Args:
            inputs: (batch_size, seq_len)
            sent_embs: (batch_size, num_sentence, sentence_embedding_dim)
        Outputs:
            output: (batch_size, seq_len, tagset_size)
        """
        batch_size = inputs.size(0)

        embeds = self.word_embeds(inputs) # (batch_size, seq_len, embedding_dim)
        embeds = embeds.transpose(0, 1) # (seq_len, batch_size, embedding_dim)
        lstm1_out, _ = self.lstm_1(self.dropout(embeds)) # (seq_len, batch_size, 2*lstm1_units)
        lstm1_out = lstm1_out.transpose(0, 1) # (batch_size, seq_len, 2*lstm1_units)
        if sent_embs is not None:
            attention_out, _ = self.att(lstm1_out, sent_embs) # (batch_size, seq_len, 2*2*lstm1_units)
        else:
            attention_out, _ = self.att(lstm1_out, lstm1_out)

        attention_out = attention_out.transpose(1, 0) # (seq_len, batch_size, 2*2*lstm1_units)
        lstm2_out, _ = self.lstm_2(attention_out) # (seq_len, batch_size, 2*lstm2_units)
        lstm2_out = lstm2_out.transpose(1, 0) # (batch_size, seq_len, 2*lstm2_units)
        
        lstm2_out = lstm2_out.contiguous()
        lstm2_out = lstm2_out.view(-1, 2*self.lstm2_units) # (batch_size*seq_len, 2*lstm2_units)
        lstm_feats = self.hidden2tag(lstm2_out) # (batch_size*seq_len, tagset_size)

        return lstm_feats.view(batch_size, -1, self.tagset_size) # (batch_size, seq_len, tagset_size)

    def _score_sentence(self, feats, tags):
        transition_score = self.crf.transition_score(tags)
        lstm_score = self._lstm_score(feats, tags)
        return transition_score + lstm_score
    
    def _lstm_score(self, feats, tags):
        mask = (tags != self.tag_to_ix[self.PAD_TAG]).float()
        tags = tags.unsqueeze(-1)
        score = torch.gather(feats, 2, tags).squeeze(-1)
        score = mask * score

        return score.sum(-1)

    def neg_log_likelihood(self, inputs, sent_embs, targets):
        """
        Args:
            inputs: (batch_size, seq_len)
            targets: (batch_size, seq_len)
            ignore_index: int
        Outputs:
            loss: Negative log likelihood loss
        """
        feats = self._get_lstm_features(inputs, sent_embs)
        losses =  self.crf.forward_alg(feats, targets) - self._score_sentence(feats, targets)

        return losses.mean()
