import argparse
import numpy as np
import pandas as pd
import json
import torch
import tensorboardX as tbx
from tensorflow.keras.preprocessing.sequence import pad_sequences
from time import time

from models import BiLM, Att_BiLSTM_CRF
from losses import SoftmaxLoss
import utils


class BatchGenerator(object):
    def __init__(self):
        self.section_embs_dict = {}
            
    def generator(self, sentences, sentembs_hash, tag_seq, batch_size):
        data_size = len(sentences)
        num_batches = np.ceil(data_size / batch_size).astype(np.int)

        shuffle_indices = np.random.permutation(np.arange(data_size))
        _sentences = np.array(sentences)[shuffle_indices]
        _sentembs_hash = np.array(sentembs_hash)[shuffle_indices]
        _tag_seq = np.array(tag_seq)[shuffle_indices]
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            batch_sentences = _sentences[start_index:end_index]
            batch_sentembs_hash = _sentembs_hash[start_index:end_index]
            batch_tag_seq = _tag_seq[start_index:end_index]

            sentence_inputs = torch.tensor(pad_sequences(batch_sentences, padding='post')).long()
            sentemb_inputs = self.pad_sentembs([self.section_embs_dict[_hash] for _hash in batch_sentembs_hash])
            outputs = torch.tensor(pad_sequences(batch_tag_seq, padding='post')).long()

            if DEVICE.type != "cpu":
                sentence_inputs = sentence_inputs.cuda()
                sentemb_inputs = sentemb_inputs.cuda()
                outputs = outputs.cuda()

            yield sentence_inputs, sentemb_inputs, outputs

    def pad_sentembs(self, sent_embs):
        max_len = len(max(sent_embs, key=len))
        return torch.stack(
            [torch.cat((embs, torch.zeros(((max_len - embs.size(0)), 200)))) for embs in sent_embs]
        )

    def get_section_embs(self, data):
        for (_id, section), g in data.groupby(['_id', 'h2']):
            self.section_embs_dict[_id + section] = torch.stack(g.sentence_emb.tolist())


def main(args):
    train_df = pd.read_pickle(args.train_data)
    valid_df = pd.read_pickle(args.valid_data)
    train_df.fillna('NO_SUBTITLE', inplace=True)
    valid_df.fillna('NO_SUBTITLE', inplace=True)
    tokenizer = get_tokenizer(args.transfer
                              , train_df.repl_words.tolist() + valid_df.repl_words.tolist())

    model = Att_BiLSTM_CRF(vocab_size=len(tokenizer.vocab_word)
                           , tag_to_ix=tokenizer.vocab_tag
                           , embedding_dim=args.word_emb_size
                           , lstm1_units=args.lstm1_units
                           , lstm2_units=args.lstm2_units)

    if args.transfer:
        bilm_model = BiLM(embedding_dim=args.bilm_emb_size
                          , lstm_units=args.bilm_lstm_units
                          , vocab_size=len(tokenizer.vocab_word))
        model = transfer_weight(model, bilm_model, args.bilm_model_path)

    # choice CPU / GPU mode
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs.")
        model = torch.nn.DataParallel(model)
    elif torch.cuda.device_count() == 1:
        print("Use single GPU.")
    else:
        print("Use CPU.")
    
    model.to(DEVICE)
    
    print("Creating data...")
    # create training data
    train_sentences = tokenizer.transform_word(train_df.repl_words.tolist())
    train_sentembs_hash = train_df.apply(lambda x: x._id + x.h2, axis=1).tolist()
    train_tag_seq = tokenizer.transform_tag(train_df.production_tag_seq.tolist())

    # create valid data
    valid_sentences = tokenizer.transform_word(valid_df.repl_words.tolist())
    valid_sentembs_hash = valid_df.apply(lambda x: x._id + x.h2, axis=1).tolist()
    valid_tag_seq = tokenizer.transform_tag(valid_df.production_tag_seq.tolist())

    # create mini-batch generator
    batch_generator = BatchGenerator()
    batch_generator.get_section_embs(train_df)
    batch_generator.get_section_embs(valid_df)

    print("Start training...")
    train(model
          , (train_sentences, train_sentembs_hash, train_tag_seq)
          , (valid_sentences, valid_sentembs_hash, valid_tag_seq)
          , epochs=args.epochs
          , batch_size=args.batch_size
          , batch_generator=batch_generator)

    print("Save model")
    torch.save(model.state_dict(), args.output)

def train(model, train_data, valid_data, epochs, batch_size, batch_generator):
    writer = tbx.SummaryWriter()

    train_sentences, train_sentembs_hash, train_tag_seq = train_data
    valid_sentences, valid_sentembs_hash, valid_tag_seq = valid_data
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        start = time()

        # training phase
        model.train()
        train_losses = []
        for sentence_inputs, sentemb_inputs, tags in \
                batch_generator.generator(train_sentences, train_sentembs_hash, train_tag_seq, batch_size):
            model.zero_grad()
            if isinstance(model, torch.nn.DataParallel):
                loss = model.module.neg_log_likelihood(sentence_inputs, sentemb_inputs, tags)
            else:
                loss = model.neg_log_likelihood(sentence_inputs, sentemb_inputs, tags)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())

        # validation phase
        model.eval()
        valid_losses = []
        for sentence_inputs, sentemb_inputs, tags in \
                batch_generator.generator(valid_sentences, valid_sentembs_hash, valid_tag_seq, batch_size):
            with torch.no_grad():
                if isinstance(model, torch.nn.DataParallel):
                    loss = model.module.neg_log_likelihood(sentence_inputs, sentemb_inputs, tags)
                else:
                    loss = model.neg_log_likelihood(sentence_inputs, sentemb_inputs, tags)
            
            valid_losses.append(loss.item())

        end = time()

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        writer.add_scalar('train_loss', train_loss, global_step=(epoch + 1))
        writer.add_scalar('valid_loss', valid_loss, global_step=(epoch + 1))
        print("Epoch {0} \t train loss: {1} \t valid loss: {2} \t exec time: {3}s".format((epoch + 1), train_loss, valid_loss, end - start))

    writer.close()

    return model

def get_tokenizer(is_transfer, sentences=None):
    tokenizer = utils.Tokenizer()

    tokenizer.vocab_tag = {
        '<PAD>': 0
        , 'B': 1
        , 'I': 2
        , 'O': 3
        , '<START>': 4
        , '<STOP>': 5
    }

    if is_transfer:
        with open("../data/all_word_vocab.json", 'r') as f:
            tokenizer.vocab_word = json.load(f)
    else:
        tokenizer.fit_word(sentences)

    return tokenizer

def transfer_weight(model, src_model, src_model_path):
    param = torch.load(src_model_path, map_location=DEVICE)
    src_model.load_state_dict(param)
    model.lstm_1.load_state_dict(src_model.bi_lstm.state_dict())
    model.word_embeds.load_state_dict(src_model.embedding.state_dict())
    return model

if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--bilm_model_path', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--word_emb_size', default=100, type=int)
    parser.add_argument('--lstm1_units', default=100, type=int)
    parser.add_argument('--lstm2_units', default=200, type=int)
    parser.add_argument('--transfer', default=True, type=bool)
    parser.add_argument('--bilm_emb_size', default=100, type=int)
    parser.add_argument('--bilm_lstm_units', default=100, type=int)

    args = parser.parse_args()

    main(args)
