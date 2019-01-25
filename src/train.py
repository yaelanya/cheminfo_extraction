import argparse
import numpy as np
import pandas as pd
import math
import torch
from tqdm import tqdm
import tensorboardX as tbx
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models import BiLM
from losses import SoftmaxLoss
import utils

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer, sentences_idx = sentence_preprocessing(args.train_data)
    bi_lm_model = BiLM(args.word_emb_size, args.lstm_unit_size, len(tokenizer.vocab_word))
    
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs.")
        bi_lm_model = torch.nn.DataParallel(bi_lm_model)
    elif torch.cuda.device_count() == 1:
        print("Use single GPU.")
    else:
        print("Use CPU.")

    bi_lm_model.to(device)
    bi_lm_model = train(bi_lm_model, sentences_idx, args.epochs, args.batch_size)
    
    torch.save(bi_lm_model.state_dict(), args.output)

def train(model, data, epochs, batch_size):
    writer = tbx.SummaryWriter()

    num_batches = math.ceil(len(data) / batch_size)

    loss_func = SoftmaxLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, minibatch, target in tqdm(batch_generator(data, batch_size)):
            optimizer.zero_grad()
            forward_output, backword_output, c = model(minibatch)
            loss = loss_func(forward_output, backword_output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data
        
        epoch_loss /= num_batches
        writer.add_scalar('epoch_loss', epoch_loss, global_step=epoch)
        print("Epoch", (i + 1), ":", epoch_loss)
        
    writer.close()

    return model

def sentence_preprocessing(filepath):
    wiki_df = pd.read_pickle(filepath)
    sentences = wiki_df.repl_words.tolist()

    tokenizer = utils.Tokenizer()
    tokenizer.fit_word(sentences)
    sentences = utils.attach_BOS_EOS(sentences)
    sentences_idx = tokenizer.transform_word(sentences)

    return tokenizer, sentences_idx

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
        batch_X = batch_data.transpose(1, 0).view(-1, batch_data.shape[0])
        
        yield (batch_num + 1), batch_X, batch_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument('--train_data', default="../data/all_wiki_sentence_split_words_using_compound_dict.pkl")
    parser.add_argument('--output', default="../model/model.h5")
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--word_emb_size', default=100)
    parser.add_argument('--lstm_unit_size', default=100)

    args = parser.parse_args()

    main(args)
