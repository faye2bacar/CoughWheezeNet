# Cough+Wheeze Net (16 kHz, PyTorch Lightning)

Pipeline complet d'entraînement et d'export mobile (TFLite) pour :
- **Détection de toux et sifflements respiratoires**
- **Classification** : {asthme, BPCO, sain}
- **Score d’exacerbation temporel**
- **VAD toux (mini-CNN)** pour filtrer les segments pertinents
- **Calcul log-mel on-device (Kotlin)** pour l’inférence en temps réel

⚠️ **Usage** : ce projet est destiné à la recherche / preuve de concept. Il ne constitue pas un outil de diagnostic médical.

---

## Arborescence

# Cough+Wheeze Net (16 kHz, PyTorch Lightning)

Pipeline d'entraînement et export mobile (TFLite) pour la détection toux/sifflement et le classement {asthme, BPCO, sain} + score d'exacerbation.

## Arborescence
```
CoughWheezeNet/
├─ README.md
├─ instructions.md
├─ env.yaml
├─ jobs/
│ └─ train-crnn.yaml
├─ src/
│ ├─ init.py
│ ├─ config.py
│ ├─ data.py
│ ├─ features.py
│ ├─ vad_cough.py
│ ├─ wheeze_feats.py
│ ├─ model_crnn.py
│ ├─ vad_model.py
│ ├─ vad_train.py
│ ├─ vad_export_tflite.py
│ ├─ train.py
│ ├─ export_tflite.py
│ └─ utils.py
├─ artifacts/
│ └─ export/
│ ├─ crnn.tflite
│ ├─ vad_cnn.tflite
│ └─ manifest.json
├─ android/
│ ├─ app/
│ │ ├─ build.gradle.kts
│ │ └─ src/main/
│ │ ├─ AndroidManifest.xml
│ │ ├─ java/com/example/coughwheeze/
│ │ │ ├─ MainActivity.kt
│ │ │ ├─ audio/AudioRecorder.kt
│ │ │ ├─ audio/MelFeature.kt
│ │ │ ├─ ml/ModelManifest.kt
│ │ │ ├─ ml/ModelUpdater.kt
│ │ │ ├─ ml/TFLiteRunner.kt
│ │ │ └─ ml/VadRunner.kt
│ │ └─ res/values/strings.xml
│ └─ gradle.properties
├─ tests/
│ └─ smoke_test.py
└─ dataset/ (exemple de données factices)
├─ audio/patient_0001/sess_01.wav
├─ audio/patient_0001/sess_02.wav
├─ audio/patient_0002/sess_01.wav
└─ labels/labels.csv
```

```
label: 0 = asthme, 1 = BPCO, 2 = sain

exac: score optionnel (0–1)
```

## Entraînement rapide (local)

```bash
python -m src.train --label_csv ./labels/labels.csv --epochs 5
```

## Export TFLite

```bash
python -m src.export_tflite --ckpt artifacts/last.ckpt --out_dir artifacts/export
```

## VAD (Voice Activity Detection) Toux | Entraîner le modèle VAD mini-CNN

```bash
python -m src.vad_train --train_csv ./labels/vad_labels.csv --epochs 10
python -m src.vad_export_tflite --out_dir artifacts/export
```

```
label : 1 = toux, 0 = non-toux
```

## Démarrage rapide
```bash
conda env create -f env.yaml && conda activate coughwheeze
python -m src.train --label_csv ./labels/labels.csv --epochs 5
python -m src.export_tflite --ckpt artifacts/last.ckpt --out_dir artifacts/export
```

## Exemple : Génération de WAV factices (toux, BPCO, sain)
```bash
import numpy as np
import soundfile as sf
import os

sr = 16000   # échantillonnage
dur = 10     # durée (s)

os.makedirs("dataset/audio/patient_0001", exist_ok=True)

# Générer bruit blanc (sain)
x = np.random.randn(sr*dur) * 0.01
sf.write("dataset/audio/patient_0001/sess_01.wav", x, sr)

# Générer toux simulée (pics courts sur bruit blanc)
x = np.random.randn(sr*dur) * 0.01
cough_pos = np.random.randint(sr, sr*(dur-1), size=5)
for p in cough_pos:
    x[p:p+2000] += np.hanning(2000) * 0.5
sf.write("dataset/audio/patient_0002/sess_01.wav", x, sr)

# Générer respiration sifflante (asthme : sinus 1–2 kHz ajouté)
t = np.arange(sr*dur) / sr
wheeze = 0.05 * np.sin(2*np.pi*1500*t)
x = np.random.randn(sr*dur) * 0.01 + wheeze
sf.write("dataset/audio/patient_0003/sess_01.wav", x, sr)
```

## Données factices générées
- Fichiers créés sous `audio/patient_0001` à 16 kHz:
  - `audio/patient_0001/sess_toux.wav` (toux simulée)
  - `audio/patient_0001/sess_bpco.wav` (BPCO simulée: souffle + sifflement)
  - `audio/patient_0001/sess_sain.wav` (sain: bruit faible)
- `labels/labels.csv` mis à jour pour pointer vers les classes:
  - `audio/patient_0001/sess_bpco.wav,1,0001,1.0,5.0,0.9`
  - `audio/patient_0001/sess_sain.wav,2,0001,0.0,3.0,0.0`

## Démarrage rapide (min, CPU, sans MLflow)
- Setup venv + deps (Windows PowerShell):
  - `pwsh -File scripts/setup_venv.ps1`
- Générer les WAV factices patient_0001:
  - `.venv\Scripts\python scripts/gen_dummy_wavs.py`
- Entraînement minimal (1 époque) sur `labels/labels.csv`:
  - `.venv\Scripts\python -m scripts.train_min --label_csv labels/labels.csv --epochs 1 --batch_size 2`
- Lancer les tests:
  - `.venv\Scripts\python -m pytest -q`

## Entraînement complet (Lightning, option MLflow)
- Sans MLflow (CPU):
  - `.venv\Scripts\python -m src.train --label_csv labels/labels.csv --epochs 5 --batch_size 4 --num_workers 0`
- Avec MLflow (si installé):
  - `.venv\Scripts\python -m pip install mlflow`
  - `.venv\Scripts\python -m src.train --label_csv labels/labels.csv --epochs 5 --batch_size 4 --num_workers 0 --mlflow`

