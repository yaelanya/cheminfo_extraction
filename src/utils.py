
def attach_BOS_EOS(sentences):
    _sents = sentences.copy()
    for s in _sents:
        s.insert(0, '<BOS>')
        s.append('<EOS>')
    
    return _sents


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