import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences


class EarlyStopping(object):
    def __init__(self, patience=5):
        self.best_loss_score = np.Inf
        self.patience = patience
        self.counter = 0
        self.stop = False
    
    def __call__(self, model, val_loss):
        if val_loss < self.best_loss_score:
            print("save best model.")
            self.best_loss_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), "checkpoint.pt")
        
    def is_stop(self):
        if self.counter >= self.patience:
            return True
        else:
            return False

class BatchGenerator(object):
    def __init__(self, batch_size=32, shuffle=True):
        self.section_embs_dict = {}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
    def generator(self, sentences, sentembs_hash, tag_seq):
        data_size = len(sentences)
        num_batches = np.ceil(data_size / self.batch_size).astype(np.int)

        if self.shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            _sentences = np.array(sentences)[shuffle_indices]
            _sentembs_hash = np.array(sentembs_hash)[shuffle_indices]
            _tag_seq = np.array(tag_seq)[shuffle_indices]
        else:
            _sentences = np.array(sentences)
            _sentembs_hash = np.array(sentembs_hash)
            _tag_seq = np.array(tag_seq)

        for batch_num in range(num_batches):
            start_index = batch_num * self.batch_size
            end_index = min((batch_num + 1) * self.batch_size, data_size)

            batch_sentences = _sentences[start_index:end_index]
            batch_sentembs_hash = _sentembs_hash[start_index:end_index]
            batch_tag_seq = _tag_seq[start_index:end_index]

            sentence_inputs = torch.tensor(pad_sequences(batch_sentences, padding='post')).long()
            sentemb_inputs = self.pad_sentembs([self.section_embs_dict[_hash] for _hash in batch_sentembs_hash])
            outputs = torch.tensor(pad_sequences(batch_tag_seq, padding='post')).long()

            yield sentence_inputs.to(self.device), sentemb_inputs.to(self.device), outputs.to(self.device)

    def pad_sentembs(self, sent_embs):
        max_len = len(max(sent_embs, key=len))
        return torch.stack(
            [torch.cat((embs, torch.zeros(((max_len - embs.size(0)), 200)))) for embs in sent_embs]
        )

    def get_section_embs(self, data):
        for (_id, section), g in data.groupby(['_id', 'h2']):
            self.section_embs_dict[_id + section] = torch.stack(g.sentence_emb.tolist())


class BatchGeneratorWithUnderSampling(BatchGenerator):
    def __init__(self, tag_to_ix, batch_size=32, shuffle=True, negative_rate=1.0):
        self.section_embs_dict = {}
        self.vocab_tag = tag_to_ix
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.negative_rate = negative_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generator(self, sentences, sentembs_hash, tag_seq):
        data_size = len(sentences)
        if self.shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            _sentences = np.array(sentences)[shuffle_indices]
            _sentembs_hash = np.array(sentembs_hash)[shuffle_indices]
            _tag_seq = np.array(tag_seq)[shuffle_indices]
        else:
            _sentences = np.array(sentences)
            _sentembs_hash = np.array(sentembs_hash)
            _tag_seq = np.array(tag_seq)

        ids_isin_BI_tag, ids_all_other_tag = self._get_ids(_tag_seq)
        positive_size = len(ids_isin_BI_tag)
        num_batches = np.ceil(positive_size * (1 + self.negative_rate) / self.batch_size).astype(np.int)
        positive_batch_size = np.ceil(positive_size / num_batches).astype(np.int)
        for batch_num in range(num_batches):
            positive_batch = \
            self._get_positive_batch((_sentences[ids_isin_BI_tag], _sentembs_hash[ids_isin_BI_tag], _tag_seq[ids_isin_BI_tag])
                                     , positive_batch_size
                                     , batch_num)

            negative_batch = \
            self._get_negative_batch((_sentences[ids_all_other_tag], _sentembs_hash[ids_all_other_tag], _tag_seq[ids_all_other_tag])
                                     , self.batch_size - positive_batch_size)

            batch_sentences, batch_sentembs_hash, batch_tag_seq = self._shuffle_batch(positive_batch, negative_batch)
            
            sentence_inputs = torch.tensor(pad_sequences(batch_sentences, padding='post')).long()
            sentemb_inputs = self.pad_sentembs([self.section_embs_dict[_hash] for _hash in batch_sentembs_hash])
            outputs = torch.tensor(pad_sequences(batch_tag_seq, padding='post')).long()

            yield sentence_inputs.to(self.device), sentemb_inputs.to(self.device), outputs.to(self.device)
    
    def _shuffle_batch(self, batch1, batch2):
        cat_batch = [np.append(data1, data2)for data1, data2 in zip(batch1, batch2)]
        sentences, sentembs_hash, tag_seq = cat_batch[0], cat_batch[1], cat_batch[2]
        shuffle_ids = np.random.permutation(range(len(sentences)))
        return sentences[shuffle_ids], sentembs_hash[shuffle_ids], tag_seq[shuffle_ids]
        
    def _get_negative_batch(self, selected_data, sample_size):
        """
        Args:
            selected_data: data of all other-tag
            sample_size: sample size of negative batch
        """
        sentences, sentembs_hash, tag_seq = selected_data
        ids = list(range(len(sentences)))
        sample_ids = np.random.choice(ids, size=sample_size, replace=False)
        return sentences[sample_ids], sentembs_hash[sample_ids], tag_seq[sample_ids]

    def _get_positive_batch(self, selected_data, sample_size, batch_num):
        """
        Args:
            selected_data: data including B or I tag
            sample_size: sample size of positive batch
            batch_num: current mini-batch number
        """
        sentences, sentembs_hash, tag_seq = selected_data

        start_index = batch_num * sample_size
        end_index = min((batch_num + 1) * sample_size, len(sentences))

        return sentences[start_index:end_index], sentembs_hash[start_index:end_index], tag_seq[start_index:end_index]

    def _get_ids(self, tag_seq):
        ids = list(range(len(tag_seq)))
        ids_isin_BI_tag = [i for i, tags in enumerate(tag_seq) \
                            if (self.vocab_tag['B'] in tags) or (self.vocab_tag['I'] in tags)]
        ids_all_other_tag = list(set(ids) - set(ids_isin_BI_tag))

        return ids_isin_BI_tag, ids_all_other_tag


class Tokenizer(object):
    def __init__(self):
        self.PAD = '<PAD>'
        self.BOS = '<BOS>'
        self.EOS = '<EOS>'
        self.UNK = '<UNK>'

        self.vocab_word = {
            self.PAD: 0
            , self.BOS: 1
            , self.EOS: 2
            , self.UNK: 3
        }
        self.vocab_char = {self.PAD: 0, self.UNK: 1}
        self.vocab_tag = {self.PAD: 0}
    
    def fit_word(self, sentences):
        for s in sentences:
            for w in s:
                if w in self.vocab_word:
                    continue
                self.vocab_word[w] = len(self.vocab_word)
                
    def fit_char(self, sentences):
        for s in sentences:
            for w in s:
                for c in w:
                    if c in self.vocab_char:
                        continue
                    self.vocab_char[c] = len(self.vocab_char)
                    
    def fit_tag(self, tag_seq):
        for tags in tag_seq:
            for tag in tags:
                if tag in self.vocab_tag:
                    continue
                self.vocab_tag[tag] = len(self.vocab_tag)
                
    def transform_word(self, sentences):
        seq = []
        for s in sentences:
            word_ids = [self.vocab_word.get(w, self.vocab_word[self.UNK]) for w in s]
            seq.append(word_ids)
            
        return seq
    
    def transform_char(self, sentences):
        seq = []
        for s in sentences:
            char_seq = []
            for w in s:
                char_ids = [self.vocab_char.get(c, self.vocab_char[self.UNK]) for c in w]
                char_seq.append(char_ids)
            seq.append(char_seq)
            
        return seq
    
    def transform_tag(self, tag_seq):
        seq = []
        for tags in tag_seq:
            tag_ids = [self.vocab_tag[tag] for tag in tags]
            seq.append(tag_ids)

        return seq

    def inverse_transform_tag(self, tag_id_seq):
        seq = []
        inv_vocab_tag = {v: k for k, v in self.vocab_tag.items()}
        for tag_ids in tag_id_seq:
            tags = [inv_vocab_tag[tag_id] for tag_id in tag_ids]
            seq.append(tags)

        return seq