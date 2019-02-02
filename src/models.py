import torch
import torch.nn as nn

class BiLM(nn.Module):
    def __init__(self, embedding_dim, lstm_units, vocab_size):
        super(BiLM, self).__init__()

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
        inputs = inputs.transpose(0, 1) # (seq_len, batch_size)
        embs = self.embedding(inputs)
        output, (h, c) = self.bi_lstm(embs)
        forward_output, backword_output = output[:, :, :self.lstm_units], output[:, :, self.lstm_units:]
        
        # shape: (batch_size * timesteps, lstm_units)
        forward_mask, backward_mask = self._get_mask(inputs)
        forward_output = forward_output[forward_mask]
        backword_output = backword_output[backward_mask]
        
        # Log-softmax
        forward_output = self._calc_proba(forward_output)
        backword_output = self._calc_proba(backword_output)
        
        return forward_output, backword_output, c

    def _calc_proba(self, embeddings):
        return nn.functional.log_softmax(self.linear(embeddings), dim=-1)
    
    def _get_mask(self, inputs):
        forward_mask = torch.cat((inputs[1:], inputs[:1])) > 2
        backward_mask = torch.cat((inputs[-1:], inputs[:-1])) > 2
        
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


class Att_BiLSTM_CRF(nn.Module):
    """
    reference: https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    """
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, lstm1_units, lstm2_units):
        super(Att_BiLSTM_CRF, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.PAD_TAG = "<PAD>"
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"

        self.embedding_dim = embedding_dim
        self.lstm1_units = lstm1_units
        self.lstm2_units = lstm2_units
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm_1 = nn.LSTM(embedding_dim, lstm1_units, num_layers=1, bidirectional=True)
        self.att = Attention(2*lstm1_units)
        self.lstm_2 = nn.LSTM(2*2*lstm1_units, lstm2_units, num_layers=1, bidirectional=True)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(2*lstm2_units, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[self.START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[self.STOP_TAG]] = -10000

    def forward(self, inputs, sent_embs):
        """
        Args:
            inputs: (batch_size, seq_len)
            sent_feats: (batch_size, num_sentence, embedding_dim)
        """
        lstm_feats = self._get_lstm_features(inputs, sent_embs)
        tag_seq_batch = []
        for feats in lstm_feats:
            _, tag_seq = self._viterbi_decode(feats)
            tag_seq_batch.append(tag_seq)

        return torch.tensor(tag_seq_batch).to(self.device.type)  # (batch_size, seq_len)

    def _get_lstm_features(self, inputs, sent_embs):
        batch_size = inputs.size(0)

        embeds = self.word_embeds(inputs) # (batch_size, seq_len, embedding_dim)
        embeds = embeds.transpose(0, 1) # (seq_len, batch_size, embedding_dim)
        lstm1_out, _ = self.lstm_1(embeds) # (seq_len, batch_size, 2*lstm1_units)
        lstm1_out = lstm1_out.transpose(0, 1)
        if sent_embs is not None:
            attention_out, _ = self.att(lstm1_out, sent_embs) # (seq_len, batch_size, 2*2*lstm1_units)
        else:
            attention_out, _ = self.att(lstm1_out, lstm1_out)
        lstm2_out, _ = self.lstm_2(attention_out) # (seq_len, batch_size, 2*lstm2_units)

        lstm2_out = lstm2_out.view(-1, 2*self.lstm2_units) # (seq_len*batch_size, 2*lstm2_units)
        lstm_feats = self.hidden2tag(lstm2_out) # (seq_len*batch_size, tagset_size)

        return lstm_feats.view(-1, batch_size, self.tagset_size).transpose(1, 0)

    def _forward_alg(self, feats, tags):
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

    def _score_sentence(self, feats, tags):
        transition_score = self._transition_score(tags)
        lstm_score = self._lstm_score(feats, tags)
        return transition_score + lstm_score

    def _transition_score(self, tags):
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
    
    def _lstm_score(self, feats, tags):
        mask = (tags != self.tag_to_ix[self.PAD_TAG]).float()
        tags = tags.unsqueeze(-1)
        score = torch.gather(feats, 2, tags).squeeze(-1)
        score = mask * score

        return score.sum(-1)

    def _viterbi_decode(self, feats):
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

    def neg_log_likelihood(self, inputs, sent_embs, targets):
        """
        Args:
            inputs: (batch_size, seq_len)
            targets: (batch_size, seq_len)
            ignore_index: int
        """
        feats = self._get_lstm_features(inputs, sent_embs)
        losses = self._score_sentence(feats, targets) - self._forward_alg(feats, targets)

        return -losses.mean()
