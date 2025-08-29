# Cough+Wheeze Net (16 kHz, PyTorch Lightning)

Pipeline d'entraînement et export mobile (TFLite) pour la détection toux/sifflement et le classement {asthme, BPCO, sain} + score d'exacerbation.

## Arborescence
```
CoughWheezeNet/
├─ README.md
├─ env.yaml
├─ jobs/
│  └─ train-crnn.yaml
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ features.py
│  ├─ vad_cough.py
│  ├─ wheeze_feats.py
│  ├─ model_crnn.py
│  ├─ train.py
│  ├─ export_tflite.py
│  └─ utils.py
├─ android/
│  ├─ app/
│  │  ├─ build.gradle.kts
│  │  └─ src/main/
│  │     ├─ AndroidManifest.xml
│  │     ├─ java/com/example/coughwheeze/
│  │     │  ├─ MainActivity.kt
│  │     │  ├─ audio/AudioRecorder.kt
│  │     │  ├─ ml/ModelManifest.kt
│  │     │  ├─ ml/ModelUpdater.kt
│  │     │  └─ ml/TFLiteRunner.kt
│  │     └─ res/values/strings.xml
│  └─ gradle.properties
└─ tests/
   └─ smoke_test.py
```

## Démarrage rapide
```bash
conda env create -f env.yaml && conda activate coughwheeze
python -m src.train --label_csv ./labels/labels.csv --epochs 5
python -m src.export_tflite --ckpt artifacts/last.ckpt --out_dir artifacts/export
```
