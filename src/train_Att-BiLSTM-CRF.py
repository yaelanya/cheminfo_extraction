import argparse
import numpy as np
import pandas as pd
import json
import torch
import tensorboardX as tbx
from time import time

from models import BiLM, Att_BiLSTM_CRF
from utils import EarlyStopping, Tokenizer, BatchGeneratorWithUnderSampling, BatchGenerator


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
    
    train_sentences, train_sentembs_hash, train_tag_seq = get_data(train_df, args.target_col, tokenizer)
    valid_sentences, valid_sentembs_hash, valid_tag_seq = get_data(valid_df, args.target_col, tokenizer)
    
    # create mini-batch generator
    if args.under_sampling:
        batch_generator = BatchGeneratorWithUnderSampling(tokenizer.vocab_tag
                                                          , batch_size=args.batch_size
                                                          , shuffle=True
                                                          , negative_rate=args.under_sampling)
    else:
        batch_generator = BatchGenerator(batch_size=args.batch_size, shuffle=True)

    batch_generator.get_section_embs(train_df)
    batch_generator.get_section_embs(valid_df)

    print("Start training...")
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.early_stopping)
    else:
        early_stopping = None
    train(model
          , (train_sentences, train_sentembs_hash, train_tag_seq)
          , (valid_sentences, valid_sentembs_hash, valid_tag_seq)
          , epochs=args.epochs
          , batch_generator=batch_generator
          , early_stopping=early_stopping)

    print("Save model")
    torch.save(model.state_dict(), args.output)

def train(model, train_data, valid_data, epochs, batch_generator, early_stopping=None):
    writer = tbx.SummaryWriter()

    train_sentences, train_sentembs_hash, train_tag_seq = train_data
    valid_sentences, valid_sentembs_hash, valid_tag_seq = valid_data
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        epoch_time = time()

        # training phase
        model.train()
        train_losses = []
        for sentence_inputs, sentemb_inputs, tags in \
                batch_generator.generator(train_sentences, train_sentembs_hash, train_tag_seq):
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
                batch_generator.generator(valid_sentences, valid_sentembs_hash, valid_tag_seq):
            with torch.no_grad():
                if isinstance(model, torch.nn.DataParallel):
                    loss = model.module.neg_log_likelihood(sentence_inputs, sentemb_inputs, tags)
                else:
                    loss = model.neg_log_likelihood(sentence_inputs, sentemb_inputs, tags)
            
            valid_losses.append(loss.item())

        epoch_time = time() - epoch_time

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        writer.add_scalar('train_loss', train_loss, global_step=(epoch + 1))
        writer.add_scalar('valid_loss', valid_loss, global_step=(epoch + 1))
        print("Epoch {0} \t train loss: {1} \t valid loss: {2} \t exec time: {3}s".format((epoch + 1), train_loss, valid_loss, epoch_time))

        if early_stopping is not None:
            early_stopping(model, valid_loss)
            if early_stopping.is_stop():
                print("Early stopping.")
                model.load_state_dict(torch.load('checkpoint.pt'))
                break

    writer.close()

    return model

def get_data(data, target_col, tokenizer):
    sentences = tokenizer.transform_word(data.repl_words.tolist())
    sentembs_hash = data.apply(lambda x: x._id + x.h2, axis=1).tolist()
    tag_seq = tokenizer.transform_tag(data[target_col].tolist())
    return sentences, sentembs_hash, tag_seq

def get_tokenizer(is_transfer, sentences=None):
    tokenizer = Tokenizer()

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
    parser.add_argument('--target_col', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--early_stopping', default=None, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--under_sampling', default=None, type=int)
    parser.add_argument('--word_emb_size', default=100, type=int)
    parser.add_argument('--lstm1_units', default=100, type=int)
    parser.add_argument('--lstm2_units', default=200, type=int)
    parser.add_argument('--transfer', default=True, type=bool)
    parser.add_argument('--bilm_model_path', type=str)
    parser.add_argument('--bilm_emb_size', default=100, type=int)
    parser.add_argument('--bilm_lstm_units', default=100, type=int)

    args = parser.parse_args()

    main(args)
