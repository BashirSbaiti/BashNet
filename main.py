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
        count += 1
        pianoRoll = pianoRoll[:, 0:cutofflen]
        pianoRoll = np.reshape(pianoRoll, [1, pianoRoll.shape[1], pianoRoll.shape[0]])
        if len(pianoRolls) == 0:
            pianoRolls = pianoRoll
        else:
            pianoRolls = np.concatenate((pianoRolls, pianoRoll),
                                        axis=0)  # final pianorolls array will be (notes, Tx, song)

notes = []

print(pianoRolls.shape)

for song in range(pianoRolls.shape[0]):
    for timestep in range(pianoRolls.shape[1]):
        noteLst = []
        for notenum in range(pianoRolls.shape[2]):  # flatten so that each time step has a string of all notes being played
            if pianoRolls[song, timestep, notenum] != 0:  # note number is given by row number
                noteLst.append(str(notenum))
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
    return temp.T  # (Tx , nx)


inp = []
for i in range(pianoRolls.shape[0]):  # inp in shape (notescombos, tx, songs)
    if len(inp) == 0:
        inp = onehot(np.array(notesInt[0:cutofflen]))
        inp = inp.reshape([1, inp.shape[0], inp.shape[1]])
    else:
        add = onehot(np.array(notesInt[cutofflen * (i - 1): cutofflen * i]))
        add = add.reshape([1, add.shape[0], add.shape[1]])
        inp = np.concatenate((inp, add), axis=0)


def decodeOh(a):  # turns back into piano roll
    notesStrs = list()
    for ohVec in range(a.shape[0]):
        notesStrs.append(intToNote[np.argmax(a[ohVec, :])])
    pr = np.zeros((128, len(notesStrs)))
    for indx, str in enumerate(notesStrs):
        for st in str.split(','):
            if st == "":
                inte = 128
            else:
                inte = int(st)
                pr[inte, indx] = 60
    return pr


print(inp.shape)
pm = piano_roll_to_pretty_midi(decodeOh(inp[8, :, :]), fs=(fps / 1.25)).write("out/test2.mid")
