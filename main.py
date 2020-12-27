import os
import pretty_midi as pm
from pretty_midi.reverse_pianoroll import piano_roll_to_pretty_midi
import numpy as np
from tensorflow import keras

fps = 40  # sampling frequency, sample one every 1/fps sec

pianoRolls = []
direc = "MIDIS"
cutofflen = 4000
count = 0
for file in os.listdir(direc):
    pmObj = pm.PrettyMIDI(direc + "/" + file)
    pmObj = pmObj.instruments[0]
    pianoRoll = pmObj.get_piano_roll(fs=fps)  # a piano roll is np.array (notes, Tx)
    if pianoRoll.shape[1] >= cutofflen:
        print(file + " \t" + str(count))
        count += 1
        pianoRoll = pianoRoll[:, 0:cutofflen]
        pianoRoll = np.reshape(pianoRoll, [pianoRoll.shape[0], pianoRoll.shape[1], 1])
        if len(pianoRolls) == 0:
            pianoRolls = np.reshape(pianoRoll, [pianoRoll.shape[0], pianoRoll.shape[1], 1])
        else:
            pianoRolls = np.concatenate((pianoRolls, pianoRoll),
                                        axis=2)  # final pianorolls array will be (notes, Tx, song)

notes = []

print(pianoRolls.shape)

for song in range(pianoRolls.shape[2]):
    for c in range(pianoRolls.shape[1]):
        noteLst = []
        for r in range(pianoRolls.shape[0]):  # flatten so that each time step has a string of all notes being played
            if pianoRolls[r, c, song] != 0:  # note number is given by row number
                noteLst.append(str(r))
        noteStr = ','.join(noteLst)
        notes.append(noteStr)

c = 0
noteToInt = dict()
for notename in sorted(set(notes)):  # map every unique string in aforementioned flat list to integers
    noteToInt.update({notename: c})
    c += 1

intToNote = dict()
for key, value in noteToInt.items():  # make inverse (map int to string)
    intToNote.update({value: key})

notesInt = []
for note in notes:
    notesInt.append(noteToInt[note])  # use mapping to make notes an int array (1, Tx*songs)


def onehot(a):
    temp = np.zeros((len(intToNote), a.size), dtype="uint8")  # convert array to onehot (nx, Tx)
    for index, value in enumerate(a):
        temp[value, index] = 1
    return temp


inp = []
for i in range(42):  # inp in shape (notescombos, tx, songs)
    if len(inp) == 0:
        inp = onehot(np.array(notesInt[0:cutofflen]))
        inp = inp.reshape([inp.shape[0], inp.shape[1], 1])
    else:
        add = onehot(np.array(notesInt[cutofflen * (i - 1): cutofflen * i]))
        add = add.reshape([add.shape[0], add.shape[1], 1])
        inp = np.concatenate((inp, add), axis=2)

print(inp.shape)


def decodeOh(a):  # turns back into piano roll
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


pm = piano_roll_to_pretty_midi(decodeOh(inp[:, :, 8]), fs=fps).write("out/test2.mid")
