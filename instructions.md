# Instructions de bout en bout — Cough+Wheeze Net (Lightning 16 kHz → Android)

> Objectif: entraîner un modèle CRNN (PyTorch Lightning, 16 kHz), l’exporter en TFLite, publier les artefacts dans Azure Blob, puis intégrer l’inférence on-device dans une app Android (Kotlin + Compose) avec mise à jour de modèle via `manifest.json`.

---
## 1) Pré-requis (OS, outils de base)
1. **Git** : installez depuis https://git-scm.com/downloads
2. **Python 3.10+** et **Conda** (Anaconda/Miniconda) :  
   - Miniconda: https://docs.conda.io/en/latest/miniconda.html
3. **FFmpeg** (optionnel pour conversions audio) :
   - macOS: `brew install ffmpeg` — Ubuntu: `sudo apt install ffmpeg` — Windows: package officiel
4. **Visual Studio Build Tools** (Windows, si besoin de compilers) : https://visualstudio.microsoft.com/visual-cpp-build-tools/

---
## 2) Cloner le repo et créer l’environnement
```bash
git clone <votre-repo-github>.git
cd CoughWheezeNet
conda env create -f env.yaml
conda activate coughwheeze
```

---
## 3) Préparer les données
- Arborescence recommandée (local ou blob monté) :
  ```
  audio/
    patient_0001/sess_01.wav
    patient_0001/sess_02.wav
    patient_0002/...
  labels/labels.csv
  ```
- `labels.csv` (ex.) :
  ```csv
  filepath,label,patient_id,start_s,end_s,exac
  audio/patient_0001/sess_01.wav,0,0001,0.0,3.0,0.2
  audio/patient_0001/sess_01.wav,2,0001,5.0,8.0,0.1
  ```
  - `label` ∈ {0: asthme, 1: COPD, 2: sain}
  - `exac` = score continu optionnel par segment (0–1)

---
## 4) Entraînement local (baseline rapide)
```bash
python -m src.train --label_csv ./labels/labels.csv --epochs 5
# Checkpoints dans ./artifacts (last.ckpt)
```

---
## 5) Installation Azure CLI & extensions ML
1. **Azure CLI** : https://learn.microsoft.com/cli/azure/install-azure-cli
2. **Login CLI** :
   ```bash
   az login
   az account set --subscription "<SUBSCRIPTION_ID>"
   ```
3. **Créer ressource group + workspace ML** :
   ```bash
   az group create -n coughwheeze-rg -l westeurope
   az ml workspace create -n coughwheeze-ml -g coughwheeze-rg
   ```
4. **Créer un cluster de calcul** :
   ```bash
   az ml compute create -n gpu-cluster -t AmlCompute --min-instances 0 --max-instances 4 --size Standard_NC6 -g coughwheeze-rg -w coughwheeze-ml
   ```

---
## 6) Charger les données dans le datastore du workspace
- Ouvrez le **Portail Azure** → Azure ML workspace → **Data** → Upload `audio/` et `labels/` vers `workspaceblobstore` (ou via `azcopy`).
- Vérifiez les chemins d’entrée dans `jobs/train-crnn.yaml`.

---
## 7) Lancer l’entraînement sur Azure ML
```bash
az ml job create -f jobs/train-crnn.yaml -g coughwheeze-rg -w coughwheeze-ml
```
- Suivez les logs dans Azure ML Studio (Jobs).  
- Les artéfacts (checkpoints) seront disponibles en sortie du job.

---
## 8) Export ONNX → TFLite (local)
1. Récupérez le `last.ckpt` (depuis les artéfacts Azure ML ou local).
2. Exécutez :
   ```bash
   python -m src.export_tflite --ckpt artifacts/last.ckpt --out_dir artifacts/export
   ```
3. Résultat : `artifacts/export/crnn.tflite` (+ onnx/savedmodel temporaires).

---
## 9) Préparer le `manifest.json` et calculer le hash
1. Calculez le hash SHA-256 du modèle :
   - macOS/Linux :
     ```bash
     shasum -a 256 artifacts/export/crnn.tflite
     ```
   - Windows PowerShell :
     ```powershell
     Get-FileHash artifacts/export/crnn.tflite -Algorithm SHA256
     ```
2. Remplacez `sha256:CHANGE_ME` par `sha256:<VOTRE_HASH>` dans `artifacts/export/manifest.json`.

---
## 10) Publier modèle + manifest dans Azure Blob
1. Créez un **Storage Account** + **Container** (public en lecture ou SAS lecture) via le portail.
2. Uploadez `crnn.tflite` et `manifest.json` dans le même dossier.
3. Copiez l’URL (ou générez un SAS).

---
## 11) Android — Installation & SDK
1. **Android Studio** : https://developer.android.com/studio
2. Installez **SDK Platform** (API 34), **Build-Tools**.
3. Ouvrez `CoughWheezeNet/android` dans Android Studio.

---
## 12) App Android — Démo d’inférence TFLite
1. Ouvrez `android/app/src/main/java/.../MainActivity.kt`.
2. Remplacez les URLs `https://<blob>` par celles de votre container (où se trouvent `manifest.json` et `crnn.tflite`).
3. Run sur un appareil (Android 10+ recommandé).  
   - **Check updates** : télécharge / met à jour le modèle, vérifie le hash.  
   - **Run demo inference** : exécute une inférence avec un input factice (remplacez par vos log-mels on-device).

> ⚠️ **Réseau en main thread** : l’exemple simple utilise `URL(...).readText()`/`readBytes()` — passez en Coroutine/Worker thread en prod.

---
## 13) (Optionnel) Calcul log-mel on-device
- Implémentez une FFT (KissFFT/JTransform) + banque de filtres mel en Kotlin.  
- Respectez : SR 16 kHz, fenêtre 25 ms (400), hop 10 ms (160), 64 mels, log1p.

---
## 14) Signature & Build Release
1. Générez une clé de signature (`Build > Generate Signed Bundle/APK`).
2. Configurez `build.gradle.kts` (signingConfig si besoin).
3. **ProGuard/R8** : gardez les classes TFLite si vous minifiez.

---
## 15) Publication (bonnes pratiques Santé)
- L’app est **d’aide au suivi**, non **diagnostique**. Ajoutez : consentement, politique de confidentialité, mentions légales, opt-in pour l’upload d’audio (désactivé par défaut).
- Tests multi-appareils, batterie (duty cycle nocturne), gestion bruits.

---
## 16) Push sur GitHub
```bash
git init
git remote add origin https://github.com/<user>/<repo>.git
git add .
git commit -m "Cough+Wheeze Net: Lightning 16k + Android TFLite starter"
git push -u origin main
```

---
## 17) Roadmap technique
- VAD toux mini-CNN `vad.tflite` séparé.
- MLP “souffle fort 6 s” pour proxy DEP/VEMS.
- Mises à jour modèles via `manifest.json` versionné, checksums, rollback.
- Eval/Monitoring: agrégation 7/30 jours, calibration seuils.

---
## FAQ
- **CPU only ?** Retirez `pytorch-cuda` de `env.yaml`.  
- **Hash mismatch ?** Vérifiez que `hash` = SHA-256 de **`crnn.tflite`** exact.  
- **CORS Blob ?** Préférez URL SAS simple (GET).  
