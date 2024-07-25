### Input files structure
Audio files belonging to each class should be placed together in a folder named as the label for that class. No other labeling is necessary. Example:
```
disvoice/
├── disvoice_character.py
├── disvoice_features.md
├── TONE_AGITATED
│   ├── agitated_sample_1.wav
│   ├── agitated_sample_2.wav
│   |── agitated_sample_3.wav
|   ...
└── TONE_RELIEVED
    ├── relieved_sample_1.wav
    ├── relieved_sample_2.wav
    |── relieved_sample_3.wav
    ...
```

### Audio characteristics Extraction
- Specify the data folder names through arguments. From example above:  
`$ python disvoice_character.py --categories TONE_AGITATED TONE_RELIEVED`
- 3 types of features are *statically* extracted from each audio file: articulation, phonation, prosody
- Output file stores shared audio labels & 3 feature matrices collected independently from each feature type

### DisVoice-Specific Dependencies
- **Praat**: [obtain](https://www.fon.hum.uva.nl/praat/download_linux.html) barren server executable; create symbolic link `praat -> praat-barren` in a `PATH` visible location
- Python=3.10; NumPy=1.22.4; SciPy=1.11.4