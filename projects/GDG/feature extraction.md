Here’s a **step-by-step guide** to convert raw audio into structured inputs (spectrograms, MIDI, symbolic encodings) and align them with labels for training latent diffusion models, music transformers, or other neural architectures:

---

### **3.1 Convert Raw Audio into Usable Input Formats**

#### **A. Spectrograms (For Latent Diffusion Models)**
Convert audio to **Mel-Spectrograms** and **CQT** (Constant-Q Transform) for vision-based models.

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate_spectrograms(audio_path, sr=44100):
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Mel-Spectrogram (64-band)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # CQT Spectrogram
    cqt = np.abs(librosa.cqt(y=y, sr=sr))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    
    # Save as images or numpy arrays
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel')
    plt.savefig(f"{audio_path}_mel.png")
    plt.close()
    
    return {
        "mel_spec": mel_spec_db,
        "cqt": cqt_db
    }
```

**Use Case**:  
- Save spectrograms as images for **latent diffusion models** (e.g., Stable Diffusion for music generation).

---

#### **B. MIDI Representation (For Music Transformers)**
Extract **pitch (notes)** and **onsets** to generate MIDI-like sequences.

```python
from pretty_midi import PrettyMIDI
import mir_eval

def audio_to_midi(audio_path, sr=44100):
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Estimate pitches and onsets
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    
    # Create a MIDI object
    midi = PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Piano
    
    # Convert pitches/onsets to MIDI notes
    for onset in onsets:
        pitch = pitches[onset]
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(pitch),
            start=onset/sr,
            end=(onset+0.5)/sr  # Fixed duration (adjust as needed)
        )
        instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    midi.write(f"{audio_path}.mid")
    return midi
```

**Use Case**:  
- Train **music transformers** (e.g., OpenAI’s MuseNet) on MIDI sequences.

---

#### **C. Symbolic Encoding (REMIs Format)**
Encode **raga, taal, notes, and rhythm** into a tokenized format like [REMIs](https://arxiv.org/abs/2002.00212).

```python
def encode_to_remi(audio_path, raga, taal, bpm):
    y, sr = librosa.load(audio_path, sr=44100)
    
    # Extract notes and onsets (simplified)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    
    # REMI tokens: [RAGA_Yaman, TAAL_Teentaal, TEMPO_120, NOTE_C4, DUR_0.5, ...]
    tokens = [
        f"RAGA_{raga}",
        f"TAAL_{taal}",
        f"TEMPO_{int(bpm)}"
    ]
    
    for onset in onsets:
        pitch = librosa.hz_to_note(pitches[onset])
        tokens.extend([
            f"NOTE_{pitch}",
            "DUR_0.25"  # Fixed duration (simplified)
        ])
    
    return tokens
```

**Example Output**:  
```
["RAGA_Yaman", "TAAL_Teentaal", "TEMPO_120", "NOTE_C4", "DUR_0.25", "NOTE_D4", ...]
```

**Use Case**:  
- Train **transformer-based models** (e.g., GPT-3 for music) on token sequences.

---

### **3.2 Align Features with Labels**
#### **A. Structured Data Format (CSV/JSON)**
```python
import pandas as pd

# Example DataFrame after feature extraction
data = {
    "audio_file": ["track1.wav", "track2.wav"],
    "raga": ["Yaman", "Bhairavi"],
    "taal": ["Teentaal", "Jhaptal"],
    "tempo_bpm": [120, 85],
    "mel_spec_path": ["track1_mel.png", "track2_mel.png"],
    "midi_path": ["track1.mid", "track2.mid"],
    "remi_tokens": [["RAGA_Yaman", ...], ["RAGA_Bhairavi", ...]]
}

df = pd.DataFrame(data)
df.to_csv("saraga_processed.csv", index=False)
```

#### **B. TFRecords (For TensorFlow)**
```python
import tensorflow as tf

def serialize_example(audio_path, raga, mel_spec):
    feature = {
        'audio_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_path.encode('utf-8')])),
        'raga': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raga.encode('utf-8')])),
        'mel_spec': tf.train.Feature(float_list=tf.train.FloatList(value=mel_spec.flatten())),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# Write TFRecords
with tf.io.TFRecordWriter("saraga.tfrecord") as writer:
    for _, row in df.iterrows():
        mel_spec = np.load(row['mel_spec_path'])
        example = serialize_example(row['audio_file'], row['raga'], mel_spec)
        writer.write(example.SerializeToString())
```

---

### **Key Workflow Summary**
1. **Input Formats**:  
   - Spectrograms → Latent diffusion models.  
   - MIDI → Music transformers.  
   - REMI tokens → Symbolic AI models.  

2. **Label Alignment**:  
   - Pair features with metadata (raga, taal) in CSV/JSON/TFRecords.  

3. **Output**:  
   ```
   saraga_processed/
   ├── mel_spectrograms/  # For diffusion models
   ├── midi/              # For transformers
   └── remi_tokens.json   # For autoregressive models
   ```

---

### **Next Steps**
- **Data Splitting**: `train_test_split` for ML pipelines.  
- **Model Training**:  
  - Use spectrograms with **CNNs/VAEs**.  
  - Use MIDI/REMIs with **transformers** (e.g., Hugging Face `transformers`).  

Would you like a **ready-to-use script** for any of these steps? 🎶