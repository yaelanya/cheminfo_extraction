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

        # (batch_size, seq_len)
        return torch.tensor(tag_seq_batch)

    def _get_lstm_features(self, inputs, sent_embs):
        batch_size = inputs.size(0)

        embeds = self.word_embeds(inputs) # (batch_size, seq_len, embedding_dim)
        embeds = embeds.transpose(0, 1) # (seq_len, batch_size, embedding_dim)
        lstm1_out, _ = self.lstm_1(embeds) # (seq_len, batch_size, 2*lstm1_units)
        if sent_embs:
            attention_out, _ = self.att(lstm1_out, sent_embs) # (seq_len, batch_size, 2*2*lstm1_units)
        else:
            attention_out, _ = self.att(lstm1_out, lstm1_out)
        lstm2_out, _ = self.lstm_2(attention_out) # (seq_len, batch_size, 2*lstm2_units)

        lstm2_out = lstm2_out.view(-1, 2*self.lstm2_units) # (seq_len*batch_size, 2*lstm2_units)
        lstm_feats = self.hidden2tag(lstm2_out) # (seq_len*batch_size, tagset_size)

        return lstm_feats.view(-1, batch_size, self.tagset_size).transpose(0, 1)


    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self._log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        alpha = self._log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
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

    def _argmax(self, vec):
    # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    # Compute log sum exp in a numerically stable way for the forward algorithm
    def _log_sum_exp(self, vec):
        max_score = vec[0, self._argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def neg_log_likelihood(self, inputs, sent_embs, targets):
        """
        Args:
            inputs: (batch_size, seq_len)
            targets: (batch_size, seq_len)
        """
        feats = self._get_lstm_features(inputs, sent_embs)
        losses = [self._forward_alg(x) - self._score_sentence(x, tags) for x, tags in zip(feats, targets)]
        
        return sum(losses)