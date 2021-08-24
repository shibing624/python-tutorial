# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 字符到字符的基本seq2seq模型
# input:hello; output:ehllo


import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense

# params
epochs = 60
batch_size = 128
rnn_size = 50
num_layers = 2
encoding_embedding_size = 15
decoding_embedding_size = 15
learning_rate = 0.001
checkpoint = 'model.ckpt'
display_step = 50
source_data_path = '../data/letters_source.txt'
target_data_path = '../data/letters_target.txt'


def get_corpus(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return text


def extract_char_vocab(data):
    """
    mapping dict
    :param data:
    :return:
    """
    special_words = ['<PAD>', '<UNK>', '<BEGIN>', '<END>']
    set_chars = list(set([char for line in data.split() for char in line]))
    # add four special words to mapping dict
    indices_char = {i: c for i, c in enumerate(special_words + set_chars)}
    char_indices = {c: i for i, c in indices_char.items()}
    return indices_char, char_indices


def get_input():
    """
    input tensor
    :return:
    """
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # get target sequence maxlen
    target_sequence_len = tf.placeholder(tf.int32, (None,), name='target_sequence_len')
    target_sequence_maxlen = tf.reduce_max(target_sequence_len, name='target_sequence_maxlen')
    source_sequence_len = tf.placeholder(tf.int32, (None,), name='source_sequence_len')

    return inputs, targets, learning_rate, target_sequence_len, target_sequence_maxlen, source_sequence_len


def get_encoder_layer(input_data, rnn_size, num_layers,
                      source_sequence_len, source_vocab_size,
                      encoding_embedding_size):
    """
    encoder layer
    :param intput_data:
    :param rnn_size:
    :param num_layers:
    :param source_sequence_len:
    :param source_vocab_size:
    :param encodeing_embedding_size:
    :return:
    """
    # encoder embedding
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data,
                                                           source_vocab_size,
                                                           encoding_embedding_size)

    # RNN cell
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for i in range(num_layers)])
    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell,
                                                      encoder_embed_input,
                                                      sequence_length=source_sequence_len,
                                                      dtype=tf.float32)
    return encoder_output, encoder_state


def process_deocder_input(data, vocab_indices, batch_size):
    """
    target sequence process: add <BEGIN>, and del last <END>
    :param data:
    :param vocab_indices:
    :param batch_size:
    :return:
    """
    # cut last char
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_indices['<BEGIN>']), ending], 1)
    return decoder_input


def decoding_layer(target_char_indices, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_len, target_sequence_maxlen, encoder_state, decoder_input,
                   batch_size=128):
    """
    decode layer
    :param target_char_indices:
    :param decoding_embedding_size:
    :param num_layers:
    :param rnn_size:
    :param target_sequence_len:
    :param target_sequence_maxlen:
    :param encoder_state:
    :param decoder_input:
    :return:
    """
    # embedding
    target_vocab_size = len(target_char_indices)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # build decoder RNN cell
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for i in range(num_layers)])

    # output fc layer
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # training decoder
    with tf.variable_scope('decode'):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_len,
                                                            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=target_sequence_maxlen)
    # predict decoder, share params with training decoder
    with tf.variable_scope('decode', reuse=True):
        start_tokens = tf.tile(tf.constant([target_char_indices['<BEGIN>']], dtype=tf.int32), [batch_size],
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens,
                                                                     target_char_indices['<END>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                            impute_finished=True,
                                                                            maximum_iterations=target_sequence_maxlen)
    return training_decoder_output, predicting_decoder_output


def seq2seq(input_data, targets, lr, target_sequence_len,
            target_sequence_maxlen, source_sequence_len,
            source_vocab_size, target_vocab_size,
            encoder_embedding_size, decoder_embedding_size,
            rnn_size, num_layers, target_char_indices, batch_size=128):
    """
    seq2seq model
    :param input_data:
    :param targets:
    :param lr:
    :param target_sequence_len:
    :param target_sequence_maxlen:
    :param source_sequence_len:
    :param source_vocab_size:
    :param target_vocab_size:
    :param encoder_embedding_size:
    :param decoder_embedding_size:
    :param rnn_size:
    :param num_layers:
    :return:
    """
    print('build model...')
    # get state output of encoder
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size, num_layers,
                                         source_sequence_len, source_vocab_size,
                                         encoder_embedding_size)
    # input of decoder
    decoder_input = process_deocder_input(targets, target_char_indices, batch_size=batch_size)
    # decoder
    training_decoder_output, predicting_decoder_output = decoding_layer(target_char_indices,
                                                                        decoder_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_len,
                                                                        target_sequence_maxlen,
                                                                        encoder_state,
                                                                        decoder_input)
    return training_decoder_output, predicting_decoder_output


def pad_sentence_batch(sentence_batch, pad_int):
    """
    pad the batch sequence, make sure every batch has same sequence_length
    :param sentence_batch:
    :param pad_int:
    :return:
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    """
    get batch by generator
    :param targets:
    :param sources:
    :param batch_size:
    :param source_pad_int:
    :param target_pad_int:
    :return:
    """
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # pad sequence
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # get sentence length
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        sources_lengths = []
        for source in sources_batch:
            sources_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, sources_lengths


def train():
    source_data = get_corpus(source_data_path)
    target_data = get_corpus(target_data_path)
    print('corpus length:', len(source_data))

    # see sample data
    print(source_data.split('\n')[:10])
    print(target_data.split('\n')[:10])

    # get mapping dict
    source_indices_char, source_char_indices = extract_char_vocab(source_data)
    target_indices_char, target_char_indices = extract_char_vocab(target_data)

    # chars index
    source_indices = [[source_char_indices.get(c, source_char_indices['<UNK>']) for c in line]
                      for line in source_data.split('\n')]
    target_indices = [
        [target_char_indices.get(c, target_char_indices['<UNK>']) for c in line] + [target_char_indices['<END>']]
        for line in target_data.split('\n')]

    # see sample source indices data
    print(source_indices[:10])
    print(target_indices[:10])

    # split data to train and validation
    train_source, valid_source = source_indices[batch_size:], source_indices[:batch_size]
    train_target, valid_target = target_indices[batch_size:], target_indices[:batch_size]

    (valid_targets_batch, valid_sources_batch,
     valid_targets_lengths, valid_sources_lengths) = next(get_batches(valid_target,
                                                                      valid_source,
                                                                      batch_size,
                                                                      source_char_indices['<PAD>'],
                                                                      target_char_indices['<PAD>']))
    train_graph = tf.Graph()
    with train_graph.as_default():
        # get inputs
        input_data, targets, learning_rate, target_sequence_len, target_sequence_maxlen, source_sequence_len = get_input()
        training_decoder_output, predicting_decoder_output = seq2seq(input_data, targets,
                                                                     learning_rate, target_sequence_len,
                                                                     target_sequence_maxlen, source_sequence_len,
                                                                     len(source_char_indices), len(target_char_indices),
                                                                     encoding_embedding_size, decoding_embedding_size,
                                                                     rnn_size, num_layers,
                                                                     target_char_indices, batch_size)
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

        masks = tf.sequence_mask(target_sequence_len, target_sequence_maxlen, dtype=tf.float32, name='masks')
        with tf.name_scope('optimization'):
            # loss
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # gradient clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(1, epochs + 1):
            batches = get_batches(train_target, train_source, batch_size,
                                  source_char_indices['<PAD>'], target_char_indices['<PAD>'])
            for batch_i, (targets_batch, sources_batch, targets_length, sources_lengths) in enumerate(batches):
                _, loss = sess.run([train_op, cost],
                                   {input_data: sources_batch,
                                    targets: targets_batch,
                                    learning_rate: learning_rate,
                                    target_sequence_len: targets_length,
                                    source_sequence_len: sources_lengths})
                if batch_i % display_step == 0:
                    validation_loss = sess.run([cost],
                                               {input_data: valid_sources_batch,
                                                targets: valid_targets_batch,
                                                learning_rate: learning_rate,
                                                target_sequence_len: valid_targets_lengths,
                                                source_sequence_len: valid_sources_lengths})
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f} - Validation Loss: {:>6.3f}'.format(
                        epoch_i,
                        epochs,
                        batch_i,
                        len(train_source) // batch_size,
                        loss,
                        validation_loss[0]
                    ))
        # save model
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model trained and saved %s' % checkpoint)


def source_2_seq(text, source_char_indices):
    """
    change source data to sequence
    :param text:
    :param source_char_indices:
    :return:
    """
    sequence_len = 7
    return [source_char_indices.get(char, source_char_indices['<UNK>']) for char in text] + \
           [source_char_indices['<PAD>']] * (sequence_len - len(text))


def infer():
    source_data = get_corpus(source_data_path)
    target_data = get_corpus(target_data_path)

    # get mapping dict
    source_indices_char, source_char_indices = extract_char_vocab(source_data)
    target_indices_char, target_char_indices = extract_char_vocab(target_data)

    input_word = 'hello'
    text = source_2_seq(input_word, source_char_indices)
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('inputs:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        source_sequence_len = loaded_graph.get_tensor_by_name('source_sequence_len:0')
        target_sequence_len = loaded_graph.get_tensor_by_name('target_sequence_len:0')
        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                          target_sequence_len: [len(input_word)] * batch_size,
                                          source_sequence_len: [len(input_word)] * batch_size})[0]
    pad = source_char_indices['<PAD>']
    print('raw input:', input_word)
    print('\nSource')
    print(' Word 编号:    {}'.format([i for i in text]))
    print(' Input Words:    {}'.format(' '.join([source_indices_char[i] for i in text])))

    print('\nTarget')
    print(' Word 编号:    {}'.format([i for i in answer_logits if i != pad]))
    print(' Response Words: {}'.format(' '.join([target_indices_char[i] for i in answer_logits if i != pad])))


if __name__ == '__main__':
    # train()
    infer()
