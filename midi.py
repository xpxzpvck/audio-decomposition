import pretty_midi
import pandas as pd
import csv
import os


def midi_to_list(midi, split_pitch=58):
    """Convert a midi file to a list of note events"""

    if isinstance(midi, str):
        midi_data = pretty_midi.pretty_midi.PrettyMIDI(midi)
    else:
        midi_data = midi

    score = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            duration = note.end - start
            pitch = note.pitch
            velocity = note.velocity / 127
            hand = '"rh"' if pitch > split_pitch else '"lh"'
            score.append([start, duration, pitch, velocity, hand])
    return score


def midi_to_csv(midi_path, csv_path, split_pitch=58):
    """Convert a midi file to a csv file"""
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    score = midi_to_list(midi_data, split_pitch)

    df = pd.DataFrame(
        score, columns=["Start", "Duration", "Pitch", "Velocity", "Label"]
    )
    df.to_csv(
        csv_path,
        sep=";",
        quoting=csv.QUOTE_NONE,
        float_format="%.5f",
        index=False,
        escapechar="\\",
    )


def csv_to_midi(csv_path, midi_path):
    """Convert a CSV file to a MIDI file."""

    df = pd.read_csv(csv_path, sep=";", quotechar='"')

    midi_data = pretty_midi.PrettyMIDI()

    inst_lh = pretty_midi.Instrument(program=0, name="Left Hand")
    inst_rh = pretty_midi.Instrument(program=0, name="Right Hand")

    for _, row in df.iterrows():
        start = float(row["Start"])
        duration = float(row["Duration"])
        end = start + duration
        pitch = int(row["Pitch"])
        velocity = int(float(row["Velocity"]) * 127)
        label = row["Label"].strip('"')

        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)

        if label == "lh":
            inst_lh.notes.append(note)
        elif label == "rh":
            inst_rh.notes.append(note)

    if inst_lh.notes:
        midi_data.instruments.append(inst_lh)
    if inst_rh.notes:
        midi_data.instruments.append(inst_rh)

    midi_data.write(midi_path)


def split_midi_by_hand(midi_path, split_pitch=58):
    """Split a MIDI file into left-hand and right-hand parts."""
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    base_name = os.path.splitext(os.path.basename(midi_path))[0]
    output_dir = os.path.dirname(midi_path)

    tempo = midi_data.estimate_tempo()

    midi_lh = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    midi_rh = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    for instrument in midi_data.instruments:
        inst_lh = pretty_midi.Instrument(
            program=instrument.program, is_drum=instrument.is_drum, name=instrument.name
        )
        inst_rh = pretty_midi.Instrument(
            program=instrument.program, is_drum=instrument.is_drum, name=instrument.name
        )

        for note in instrument.notes:
            target = inst_rh if note.pitch > split_pitch else inst_lh
            target.notes.append(
                pretty_midi.Note(
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                    velocity=note.velocity,
                )
            )

        if inst_lh.notes:
            midi_lh.instruments.append(inst_lh)
        if inst_rh.notes:
            midi_rh.instruments.append(inst_rh)

    midi_lh.write(os.path.join(output_dir, f"{base_name}_lh.mid"))
    midi_rh.write(os.path.join(output_dir, f"{base_name}_rh.mid"))
