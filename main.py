import os
import pretty_midi as pm
from pretty_midi.reverse_pianoroll import piano_roll_to_pretty_midi
import numpy as np
import keras
import tensorflow as tf

fps = 40  # sampling frequency, sample one every 1/fps sec

pianoRolls = []
for file in os.listdir("midis"):
    pmObj = pm.PrettyMIDI("midis/" + file)
    pmObj = pmObj.instruments[0]
    pianoRoll = pmObj.get_piano_roll(fs=fps)  # a piano roll is np.array (notes, Tx)
    if len(pianoRolls) == 0:
        pianoRolls = pianoRoll
    else:
        pianoRolls = np.concatenate((pianoRolls, pianoRoll), axis=1)

total = 0
num = 0
notes = []

for c in range(pianoRolls.shape[1]):
    noteLst = []
    for r in range(pianoRolls.shape[0]):
        if pianoRolls[r, c] != 0:
            noteLst.append(str(r))
    noteStr = ','.join(noteLst)
    notes.append(noteStr)

c = 0
noteToInt = dict()
for notename in sorted(set(notes)):
    noteToInt.update({notename: c})
    c += 1

intToNote = dict()
for key, value in noteToInt.items():
    intToNote.update({value: key})

notesInt = []
for note in notes:
    notesInt.append(noteToInt[note])


def onehot(a):
    temp = np.zeros((a.max() + 1, a.size), dtype="uint8")
    for index, value in enumerate(a):
        temp[value, index] = 1
    return temp

inp = onehot(np.array(notesInt))


def decodeOh(a):
    notesStrs = list()
    for ohVec in range(a.shape[1]):
        notesStrs.append(intToNote[np.argmax(a[:, ohVec])])
    pr = np.zeros((128, len(notesStrs)))
    for indx, str in enumerate(notesStrs):
        for st in str.split(','):
            if st == "":
                inte = 128
            else:
                inte = int(st)
                pr[inte, indx] = 60
    return pr

pm = piano_roll_to_pretty_midi(decodeOh(inp)).write("out/test2.mid")

def create_model(seq_len, unique_notes, dropout=0.3, output_emb=100, rnn_unit=128, dense_unit=64):  # TODO: this
    inputs = keras.layers.Input(shape=(seq_len,))
    embedding = keras.layers.Embedding(input_dim=unique_notes + 1, output_dim=output_emb, input_length=seq_len)(inputs)
    forward_pass = keras.layers.Bidirectional(keras.layers.GRU(rnn_unit, return_sequences=True))(embedding)
    forward_pass, att_vector = tf.SeqSelfAttention(
        return_attention=True,
        attention_activation='sigmoid',
        attention_type=tf.SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_width=50,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        bias_regularizer=keras.regularizers.l1(1e-4),
        attention_regularizer_weight=1e-4,
    )(forward_pass)
    forward_pass = keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = keras.layers.Bidirectional(keras.layers.GRU(rnn_unit, return_sequences=True))(forward_pass)
    forward_pass, att_vector2 = tf.SeqSelfAttention(
        return_attention=True,
        attention_activation='sigmoid',
        attention_type=tf.SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_width=50,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        bias_regularizer=keras.regularizers.l1(1e-4),
        attention_regularizer_weight=1e-4,
    )(forward_pass)
    forward_pass = keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = keras.layers.Bidirectional(keras.layers.GRU(rnn_unit))(forward_pass)
    forward_pass = keras.layers.Dropout(dropout)(forward_pass)
    forward_pass = keras.layers.Dense(dense_unit)(forward_pass)
    forward_pass = keras.layers.LeakyReLU()(forward_pass)
    outputs = keras.layers.Dense(unique_notes + 1, activation="softmax")(forward_pass)

    model = keras.Model(inputs=inputs, outputs=outputs, name='generate_scores_rnn')
    return model


# model = create_model(128, 128)
