import argparse
import numpy as np
import pandas as pd
import math
import torch
import tensorboardX as tbx
from tensorflow.keras.preprocessing.sequence import pad_sequences
from time import time

from models import BiLM
from losses import SoftmaxLoss
from utils import Tokenizer, EarlyStopping


def main(args):
    train_df = pd.read_pickle(args.train_data)
    valid_df = pd.read_pickle(args.valid_data)
    tokenizer = Tokenizer()
    tokenizer.fit_word(train_df.repl_words.tolist())

    train_sentences_idx = sentence_preprocessing(train_df, tokenizer)
    valid_sentences_idx = sentence_preprocessing(valid_df, tokenizer)

    bi_lm_model = BiLM(args.word_emb_size, args.lstm_unit_size, len(tokenizer.vocab_word))
    
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs.")
        bi_lm_model = torch.nn.DataParallel(bi_lm_model)
    elif torch.cuda.device_count() == 1:
        print("Use single GPU.")
    else:
        print("Use CPU.")
    bi_lm_model.to(device)

    bi_lm_model = train(bi_lm_model
                        , train_sentences_idx
                        , valid_sentences_idx
                        , args.epochs
                        , args.batch_size
                        , args.early_stopping)
    
    torch.save(bi_lm_model.state_dict(), args.output)

def train(model, train_data, valid_data, epochs, batch_size, patience):
    writer = tbx.SummaryWriter()

    loss_func = SoftmaxLoss()
    optimizer = torch.optim.Adam(model.parameters())
    early_stopping = EarlyStopping(patience=patience)
    for epoch in range(epochs):
        start = time()
        # training phase
        model.train()
        train_losses = []
        for i, minibatch in batch_generator(train_data, batch_size):
            model.zero_grad()
            forward_output, backword_output, _ = model(minibatch)
            loss = loss_func(forward_output, backword_output, minibatch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validation phase
        model.eval()
        valid_losses = []
        for i, minibatch in batch_generator(valid_data, batch_size):
            with torch.no_grad():
                forward_output, backword_output, _ = model(minibatch)
                loss = loss_func(forward_output, backword_output, minibatch)
                valid_losses.append(loss.item())

        end = time()

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        writer.add_scalar('train_loss', train_loss, global_step=(epoch + 1))
        writer.add_scalar('valid_loss', valid_loss, global_step=(epoch + 1))
        print("Epoch {0} \t train loss: {1} \t valid loss: {2} \t exec time: {3}s".format((epoch + 1), train_loss, valid_loss, end - start))

        if patience is not None:
            early_stopping(model, valid_loss)
            if early_stopping.is_stop():
                print("Early stopping.")
                model.load_state_dict(torch.load('checkpoint.pt'))
                break

    writer.close()

    return model

def sentence_preprocessing(data, tokenizer):
    sentences = data.repl_words.tolist()
    sentences = attach_BOS_EOS(sentences)
    sentences_idx = tokenizer.transform_word(sentences)
    return sentences_idx

def attach_BOS_EOS(sentences):
    _sents = sentences.copy()
    for s in _sents:
        s.insert(0, '<BOS>')
        s.append('<EOS>')
    return _sents

def batch_generator(data, batch_size):
    data_size = len(data)
    num_batches = math.ceil(data_size / batch_size)
    
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffle_data = np.array(data)[shuffle_indices]
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        batch_data = shuffle_data[start_index:end_index]
        batch_data = pad_sequences(batch_data, padding='post')
        batch_data = torch.tensor(batch_data).long()
        
        yield (batch_num + 1), batch_data.to(device)


if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--valid_data', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--word_emb_size', default=100, type=int)
    parser.add_argument('--lstm_unit_size', default=100, type=int)
    parser.add_argument('--early_stopping', default=None, type=int)

    args = parser.parse_args()

    main(args)
