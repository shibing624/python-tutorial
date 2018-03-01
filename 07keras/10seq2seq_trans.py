'''
a basic character-level sequence to sequence model.
'''
# coding=utf-8
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

# config
batch_size = 64
epochs = 1
hidden_dim = 256
num_samples = 10000
data_path = './fra-eng/fra.txt'
save_model_path = './s2s.h5'
# vector of data
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
lines = open(data_path, 'r', encoding='utf-8').read().split("\n")
for line in lines[:min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split("\t")
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_len = max([len(text) for text in input_texts])
max_decoder_seq_len = max([len(text) for text in target_texts])

print('num of samples:', len(input_texts))
print('num of unique input tokens:', num_encoder_tokens)
print('num of unique output tokens:', num_decoder_tokens)
print('max sequence length for inputs:', max_encoder_seq_len)
print('max sequence length for outputs:', max_decoder_seq_len)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_len, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_len, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_len, num_decoder_tokens), dtype='float32')

# one hot representation
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is a head of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

# encoder decoder process
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(hidden_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# discard 'encoder_outputs' and only keep the states
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
# the decoder to return full output sequences and internal states
decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# save
model.save(save_model_path)
print('save model:', save_model_path)

# inference
# sample models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(hidden_dim,))
decoder_state_input_c = Input(shape=(hidden_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# reverse lookup token index to decode sequences back
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # encoder the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.0

    # sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # exit condition
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_len:
            stop_condition = True

        # update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # update states
        states_value = [h, c]
    return decoded_sentence


for seq_index in range(10):
    # take one sequence (part of the training set) for decoding.
    input_seq = encoder_input_data[seq_index:seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('input sentence:', input_texts[seq_index])
    print('decoded sentence:', decoded_sentence)
