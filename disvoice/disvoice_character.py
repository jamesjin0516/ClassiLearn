import argparse
from disvoice.articulation.articulation import Articulation
from disvoice.phonation.phonation import Phonation
from disvoice.prosody.prosody import Prosody
import joblib
import numpy as np
import os
import pandas as pd


parser = argparse.ArgumentParser()
# python disvoice_character.py -c HC PD    where "HC" and "PD" are names of folders storing audios of different categories
parser.add_argument("-c", "--categories", nargs="+", required=True, help="Input audios' directories, each represent one category")
args = parser.parse_args()


extractors = {"articulation": Articulation(), "phonation": Phonation(), "prosody": Prosody()}
audio_groups = args.categories
feats_tables = {group: {extr_type: None for extr_type in extractors} for group in audio_groups}


for group in audio_groups:
    for file_name in os.listdir(group):
        # Extract features of each type from all audio files 
        for extr_type in extractors:
            pending = True
            while pending:
                try:
                    feats = extractors[extr_type].extract_features_file(os.path.abspath(os.path.join(group, file_name)), static=True, fmt="csv")
                    pending = False
                except FileNotFoundError:
                    print(f"{file_name}: {extr_type} recomputing...")
            if feats_tables[group][extr_type] is None:
                feats_tables[group][extr_type] = feats
            else:
                feats_tables[group][extr_type] = pd.concat([feats_tables[group][extr_type], feats], ignore_index=True)
    # Save features of each type from all audio files in the current category
    for extr_type in feats_tables[group]:
        feats_tables[group][extr_type].to_csv(os.path.join(group, f"{group}_{extr_type}_characteristics.csv"), na_rep="0", index=False)


comb_feats = {extr_type: None for extr_type in feats_tables[list(feats_tables.keys())[0]]}
labels = []
for extr_type in comb_feats:
    groups_feats = []
    # Combine features of the same type from all clases 
    for group in feats_tables:
        with open(os.path.join(group, f"{group}_{extr_type}_characteristics.csv"), "rb") as feats_file:
            groups_feats.append(np.loadtxt(feats_file, delimiter=",", skiprows=1))
        if isinstance(labels, list):
            labels.append(np.full(groups_feats[-1].shape[0], audio_groups.index(group), dtype=int))
    comb_feats[extr_type] = np.concatenate(groups_feats)
    if isinstance(labels, list): labels = np.concatenate(labels)
joblib.dump(dict(**comb_feats, labels=labels), "combined_characteristics.gz")
