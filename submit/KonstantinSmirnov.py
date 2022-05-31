import enum
import math
from random import randint
from itertools import groupby
import numpy as np
import random as rnd
from mido import MidiFile, Message, MidiTrack
import music21
import copy
from operator import add
import pretty_midi
from collections import Counter

""" The path to the music file """
filepath = 'barbiegirl_mono.mid'

""" returns a list [note midi-number, start_time, end_time] of all notes playing"""
def parseMidi(filepath: str):
    notes_list = []
    midi_data = pretty_midi.PrettyMIDI(filepath)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes_list.append([note.pitch, note.start, note.end])
    return notes_list
""" 
Converts string representation of a note to its number in octave. 
A -> 0
B -> 1
...
"""
def midiStringToInt(note: str):
    notes = [["A"], ["A#", "Bb"], ["B"], ["C"], ["C#", "Db"], ["D"], ["D#", "Eb"], ["E"], ["F"], ["F#", "Gb"], ["G"],
             ["G#", "Ab"]]
    for i in range(len(notes)):
        if note in notes[i]:
            return i
    return None
# get tonic and mode of the melody
# E.g. C# minor
score = music21.converter.parse(filepath)
notes_list = parseMidi(filepath)
key = score.analyze('key')
print(key.tonic, key.mode)

# open midi-file
mid = MidiFile(filepath, clip=True)
# for track in mid.tracks:
#     print(track)

CHORD_PERIOD = 2 * mid.ticks_per_beat
AMOUNT_OF_CHORDS = math.ceil(mid.length)
print(f'Amount of chords: {AMOUNT_OF_CHORDS}')

""" The circle of Fifths """
chord_circle = [['C', 'A', 'B'], ['G', 'E', 'F#'], ['D', 'B', 'C#'], ['A', 'F#', 'G#'],
                ['E', 'C#', 'D#'], ['B', 'G#', 'A#'], ['F#', 'D#', 'E#'], ['C#', 'A#', 'C'],
                ['G#', 'F', 'G'], ['D#', 'C', 'D'], ['A#', 'G', 'A'], ['F', 'D', 'E']]

""" The base note of the octave in which i'm gonna make accompaniment. It is two octaves lower than the octave of the original melody is"""
baseOctaveNote = key.tonic.midi - midiStringToInt(key.tonic.name) - 12

""" 
Chord class, it serves as an array with notes.
Types of chords: major, minor, sus2, sus4
"""
class Chord:
    def __init__(self, note: int = None, type: int = 0):
        # kinds of chords I use
        major = [0, 4, 7]
        minor = [0, 3, 7]
        sus2 = [0, 2, 7]
        sus4 = [0, 5, 7]
        dim = [0, 3, 6]
        self.chord = [note, note, note]
        # major
        if type == 0:
            self.chord = list( map(add, self.chord, major))
        # minor
        elif type == 1:
            self.chord = list(map(add, self.chord, minor))
        # sus2
        elif type == 2:
            self.chord = list(map(add, self.chord, sus2))
        # sus4
        elif type == 3:
            self.chord = list(map(add, self.chord, sus4))
        # dim
        elif type == 4:
            self.chord = list(map(add, self.chord, dim))

    def __getitem__(self, key):
        return self.chord[key]


class Chromosome:
    def __init__(self, baseOctaveNote: int, genes_amount=AMOUNT_OF_CHORDS, genes=None):
        self.length = genes_amount if genes is None else len(genes)
        # convert scale to numerical form
        mode = 0 if key.mode == 'major' else 1
        ind = 0
        # find the tonic note in the circle of fifths and save its index in the circle
        for i in range(len(chord_circle)):
            if chord_circle[i][mode] == key.tonic.name:
                ind = i
                break
        # here I select the chords that are the most suitable to the melody according to the circle of fifths
        self.sus_chords = [Chord(baseOctaveNote + midiStringToInt(chord_circle[ind][mode]), type=2),
                           Chord(baseOctaveNote + midiStringToInt(chord_circle[ind][abs(1 - mode)]),
                                 type=2),
                           Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind + 1) % 12][abs(1 - mode)]),
                                 type=2),
                           Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind - 1) % 12][abs(1 - mode)]),
                                 type=2),
                           Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind + 1) % 12][mode]), type=2),
                           Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind - 1) % 12][mode]), type=2),

                           Chord(baseOctaveNote + midiStringToInt(chord_circle[ind][mode]), type=3),
                           Chord(baseOctaveNote + midiStringToInt(chord_circle[ind][abs(1 - mode)]),
                                 type=3),
                           Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind + 1) % 12][abs(1 - mode)]),
                                 type=3),
                           Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind - 1) % 12][abs(1 - mode)]),
                                 type=3),
                           Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind + 1) % 12][mode]), type=3),
                           Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind - 1) % 12][mode]), type=3)
                           ]

        self.major_minor_dim_chords = [Chord(baseOctaveNote + midiStringToInt(chord_circle[ind][mode]), type=mode),
                                Chord(baseOctaveNote + midiStringToInt(chord_circle[ind][abs(1 - mode)]),
                                      type=abs(1 - mode)),
                                Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind + 1) % 12][abs(1 - mode)]),
                                      type=abs(1 - mode)),
                                Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind - 1) % 12][abs(1 - mode)]),
                                      type=abs(1 - mode)),
                                Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind + 1) % 12][mode]), type=mode),
                                Chord(baseOctaveNote + midiStringToInt(chord_circle[(ind - 1) % 12][mode]), type=mode),

                                Chord(baseOctaveNote + midiStringToInt(chord_circle[ind][2]), type=4),
                                ]
        # create population of random chords
        if genes is None:
            population = []
            for i in range(self.length):
                population.append(self.chooseRandomChord())
            self.genes = population
        else:
            self.genes = genes
    # selects random chord. Suspended chords are kinda jazzy and don't always sound appliable to melodies
    # that's why there's a little chance of its selection
    def chooseRandomChord(self):
        if rnd.random() <= 0.05:
            rand = rnd.randint(0, len(self.sus_chords) - 1)
            return self.sus_chords[rand]
        rand = rnd.randint(0, len(self.major_minor_dim_chords) - 1)
        return self.major_minor_dim_chords[rand]

    def get_genes(self):
        return list(self.genes).copy()

    def __getitem__(self, key):
        return self.genes[key]

    def __len__(self):
        return len(self.genes)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __setitem__(self, key, value):
        self.__dict__[key] = value


# Genetic Algorithm class
class GA:
    # initialize the first population
    # i have different size of population on initialization to increase exploration of chords
    def __init__(self, init_pop_size=500, pop_size=20):
        self.pop_size = pop_size
        self.population = []
        for i in range(init_pop_size):
            self.population.append(Chromosome(baseOctaveNote))
    # for mutation I just replace randomly selected chord with a random chord
    def mutation(self, sol):
        n = len(sol)
        pos = rnd.randint(0, n - 1)
        pos_2 = rnd.randint(0, n - 1)
        # result = self.swap(sol, pos_1, pos_2)
        result = copy.copy(sol)
        result[pos] = sol.chooseRandomChord()
        result[pos_2] = sol.chooseRandomChord()
        return result
    # swap chords
    def swap(self, sol, posA, posB):
        result = copy.copy(sol)
        t = sol[posA]
        result[posA] = sol[posB]
        result[posB] = t
        return result
    # two-point crossover
    def crossover(self, solA, solB):
        n = len(solA)
        child = list(copy.copy(solA))
        num_els = np.ceil(n * (rnd.randint(10, 90) / 100))
        start_pnt = rnd.randint(0, n - 2)
        end_pnt = n if int(start_pnt + num_els) > n else int(start_pnt + num_els)
        child[start_pnt:end_pnt] = solB[start_pnt:end_pnt]
        return Chromosome(baseOctaveNote, genes=child)

    def fitness_function(self, solution):
        fitness = 0
        time = 0
        prev_note_chord_pair = [None, None]
        # for each chord
        for i in range(len(solution)):
            chord = solution[i]
            ind = 4 * i if 4 * i < len(notes_list) else len(notes_list) - 1
            # get the note playing at the begining of even beat (2k)
            note = notes_list[ind][0]
            # save current note-chord pair as previous if it's the first pair
            if i == 0:
                prev_note_chord_pair = [note, chord[0]]
            else:
                # if a note pitch is higher/lower than the previous one, it's better for the chord to be higher/lower 
                # than the previous
                if note > prev_note_chord_pair[0] and chord[0] > prev_note_chord_pair[1] or \
                        note < prev_note_chord_pair[0] and chord[0] < prev_note_chord_pair[1]:
                    fitness += 10
            # if the adjacent chords differ too much in pitch, punish
            if abs(chord[0] - prev_note_chord_pair[1]) > 5:
                fitness -= 8
            # if any note in chord is the same as the note of melody playing or is in harmonic difference, reward
            diffs = [(note - e) % 12 for e in chord]
            if 0 in diffs:
                fitness += 10
            elif (diffs[0] >= 3 and diffs[0] <= 8) or diffs[0] == 11:
                fitness += 10
            # else punish
            else:
                fitness -= 10
            threshold = 6
            border = 0 if i - threshold < 0 else i - threshold
            count = Counter(solution[border:i])
            if count[chord] > 2:
                fitness-=10
            time += 1
            prev_note_chord_pair = [note, chord[0]]
        # if the last chord has the base of the last tonic note, reward
        # usually songs end with tonic note
        if solution[-1][0] == key.tonic.midi:
            fitness+=50

        # if there are more than 2 the same chords in a row, punish
        for _, group in groupby(solution):
            count_dups = sum(1 for _ in group)
            if count_dups > 2:
                fitness = -100

        return fitness

    # pick a parent based on fitness value
    def pickSolution(self, pop_bag):
        fit_bag_evals = self.evaluatePopulationFitness(pop_bag)
        picked = fit_bag_evals["solution"][0]
        while True:
            rnIndex = rnd.randint(0, len(pop_bag) - 1)
            rnPick = fit_bag_evals["fit_wgh"][rnIndex]
            r = rnd.random()
            if r <= rnPick:
                picked = fit_bag_evals["solution"][rnIndex]
                break
        return picked

    # Evaluate fitness of population
    def evaluatePopulationFitness(self, pop_bag):
        result = {}
        fit_vals = []
        solutions = []
        for solution in pop_bag:
            fit_vals.append(self.fitness_function(solution))
            solutions.append(solution)
        # save the fitness values
        result["fit_vals"] = fit_vals
        # calculate the probabilities: the higher the fitness, the bigger its probability of selection
        fit_prob = [i - min(list(result["fit_vals"])) + 1 for i in list(result["fit_vals"])]
        result["fit_wgh"] = [i / sum(fit_prob) for i in fit_prob]
        result["solution"] = solutions
        return result

    def run(self):
        # The initial population
        pop_bag = ga.population
        best_fit_global = None
        best_solution_global = None
        # Iterating over all generations
        for g in range(2000):
            print(f'Generation {g}...')
            # Evaluate fitness for all individuals
            pop_bag_fit = ga.evaluatePopulationFitness(pop_bag)
            # Best individual so far
            best_fit = np.max(pop_bag_fit["fit_vals"])
            best_fit_index = pop_bag_fit["fit_vals"].index(best_fit)
            best_solution = pop_bag_fit["solution"][best_fit_index]
            # at first iteration there's nothing to compare to the solutions
            if g == 0:
                best_fit_global = best_fit
                best_solution_global = best_solution
            # if found new best, save it
            else:
                if best_fit >= best_fit_global:
                    best_fit_global = best_fit
                    best_solution_global = best_solution
            # Create the new population bag
            new_pop_bag = []
            for i in range(self.pop_size):
                # Pick 2 parents from the generation
                pA = ga.pickSolution(pop_bag)
                pB = ga.pickSolution(pop_bag)
                new_element = pA
                # Crossover the parents with probability
                if rnd.random() <= 0.9:
                    new_element = ga.crossover(pA, pB)
                # Mutate the child with probability
                if rnd.random() <= 0.4:
                    new_element = ga.mutation(new_element)
                # Append the child to the bag
                new_pop_bag.append(new_element)
            pop_bag = new_pop_bag
        return  best_solution_global

# save the music to midi file
def saveSolution(solution: Chromosome, path=filepath[:-4]+'-withAccompaniment.mid'):
    VELOCITY = 50
    track = MidiTrack()
    # length of a chord in ticks
    CHORD_TICKS = 2 * mid.ticks_per_beat
    for chord in solution:
        track.append(Message('note_on', note=chord[0], velocity=VELOCITY, time=0))
        track.append(Message('note_on', note=chord[1], velocity=VELOCITY, time=0))
        track.append(Message('note_on', note=chord[2], velocity=VELOCITY, time=0))

        track.append(Message('note_off', note=chord[0],
                             velocity=0, time=CHORD_TICKS))
        track.append(Message('note_off', note=chord[1],
                             velocity=0, time=0))
        track.append(Message('note_off', note=chord[2],
                             velocity=0, time=0))
    mid.tracks.append(track)
    mid.save(path)

# i have different size of population on initialization to increase exploration of chords
ga = GA(init_pop_size=500, pop_size=20)
best_solution = ga.run()
saveSolution(best_solution)
print("DONE")


