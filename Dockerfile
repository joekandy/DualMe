FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Imposta le variabili d'ambiente
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Installa le dipendenze di sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Crea e imposta la directory di lavoro
WORKDIR /workspace

# Copia i file del progetto
COPY requirements.txt .
COPY setup.py .
COPY dualme/ dualme/

# Crea la directory per i modelli e copia i modelli
RUN mkdir -p /workspace/models
COPY virtual_tryon.pth /workspace/models/upper_model.pth
COPY virtual_tryon_dc.pth /workspace/models/dressed_model.pth

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Crea le directory necessarie nel volume persistente
RUN mkdir -p /workspace/models \
    /workspace/data \
    /workspace/logs \
    /workspace/checkpoints

# Script di avvio
COPY start.sh /workspace/
RUN chmod +x /workspace/start.sh

# Espone la porta per Gradio
EXPOSE 7860

# Comando di avvio
CMD ["/workspace/start.sh"]

## Note

- I modelli di DensePose e OpenPose sono necessari per il funzionamento del sistema
- Assicurati di avere una GPU compatibile con CUDA
- Il sistema è ottimizzato per RunPod 

# DualMe Virtual Try-On

Sistema di Virtual Try-On basato su deep learning per provare virtualmente i vestiti.

## Installazione

1. Clona il repository:
```bash
git clone https://github.com/joekandy/DUALME-DEF.git
cd DUALME-DEF
```

2. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

3. Scarica i modelli necessari:
- Crea una cartella `checkpoints` nella directory principale
- Scarica i modelli da questi link:
  - [DensePose Model](https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl)
  - [OpenPose Model](https://www.dropbox.com/s/2k9wtvnw9dx1cf2/body_pose_model.pth)

4. Avvia l'applicazione:
```bash
python -m dualme.app.gradio_app
```

## Struttura del Progetto 

## Caratteristiche

- Prova virtuale di abiti
- Supporto per diversi tipi di capi (dressed, bottom, upper)
- Interfaccia Gradio user-friendly
- Ottimizzato per GPU

## Deployment su RunPod

1. Vai su [RunPod](https://www.runpod.io/)
2. Crea un nuovo pod con:
   - Template: PyTorch 2.1.0
   - GPU: NVIDIA A4000 o superiore
   - Volume: /workspace
   - Port: 7860

3. Clona il repository nel pod:
```bash
git clone https://github.com/joekandy/DUALME-DEF.git
cd DUALME-DEF
```

4. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

5. Avvia l'applicazione:
```bash
python -m dualme.app.gradio_app
```

## Note

- I modelli di DensePose e OpenPose sono necessari per il funzionamento del sistema
- Assicurati di avere una GPU compatibile con CUDA
- Il sistema è ottimizzato per RunPod 

# Rimuovi le cartelle non necessarie
rm -rf dualme/src
rm -rf dualme/virtual-try-on-main

# Crea le cartelle necessarie se non esistono
mkdir -p dualme/app
mkdir -p dualme/models
mkdir -p dualme/utils
mkdir -p dualme/configs

# Sposta i file nelle cartelle corrette
mv vto_app.py dualme/app/gradio_app.py
mv human_parser.py dualme/models/

# Rimuovi tutto dal tracking
git rm -r --cached .

# Aggiungi solo i file necessari
git add .gitignore
git add dualme/
git add requirements.txt
git add setup.py
git add Dockerfile
git add README.md

# Commit e push
git commit -m "Clean repository and reorganize structure"
git push origin main

mkdir DualMe-Clean
cd DualMe-Clean

model:
  image_size: 768
  steps: 30
  scale: 2.5
  seed: 42

app:
  host: "0.0.0.0"
  port: 7860
  share: true
  enable_queue: true

garment_types:
  - dressed
  - bottom
  - upper

paths:
  checkpoints: "checkpoints/"
  examples: "examples/"

git init
git add .
git commit -m "Initial commit: Clean DualMe repository"
git remote add origin https://github.com/joekandy/DUALME-DEF.git
git push -f origin main