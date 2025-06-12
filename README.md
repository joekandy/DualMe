# DualMe Virtual Try-On App

Applicazione di Virtual Try-On basata su DualMe, ottimizzata per il deployment su RunPod.

## 🚀 Deployment su RunPod

1. **Crea un nuovo template su RunPod**
   - Vai su [RunPod](https://www.runpod.io)
   - Seleziona "Stable Diffusion" come template base
   - Imposta CUDA 11.8.0

2. **Configura il container**
   - Copia il contenuto di questo repository nel container
   - Assicurati che il Dockerfile sia nella root
   - Imposta la porta 7860 come esposta

3. **Avvia il container**
   ```bash
   docker build -t dualme-vto-app .
   docker run --gpus all -p 7860:7860 dualme-vto-app
   ```

4. **Accedi all'interfaccia**
   - Apri il browser e vai a `http://localhost:7860`
   - L'interfaccia Gradio sarà disponibile

## 🛠️ Struttura del Progetto

```
VTO/
├── dualme/app/main.py        # Applicazione principale DualMe
├── requirements.txt         # Dipendenze Python
├── Dockerfile               # Configurazione Docker
├── models/                  # Directory modelli
│   ├── humanparsing/
│   └── ...
└── examples/                # Immagini di esempio
```

## 📝 Uso

1. Carica un'immagine della persona
2. Carica un'immagine del capo d'abbigliamento
3. (Opzionale) Modifica i parametri avanzati:
   - Steps: numero di passi di denoising (default: 30)
   - Scale: intensità dell'effetto (default: 2.5)
   - Seed: per riproducibilità (default: 42)
4. Clicca "Genera"

## ⚙️ Requisiti Hardware

- GPU NVIDIA con almeno 8GB VRAM
- 16GB RAM
- 50GB spazio disco

## 🔧 Troubleshooting

1. **Errore CUDA**
   - Verifica che la versione di CUDA sia 11.8.0
   - Controlla che i driver NVIDIA siano aggiornati

2. **Errore memoria**
   - Riduci la dimensione delle immagini
   - Diminuisci il numero di steps

3. **Errore modelli**
   - Verifica che tutti i modelli siano nella cartella `models`
   - Controlla i permessi delle cartelle

## 📞 Supporto

Per problemi o domande, apri una issue su GitHub.

## Avvio DualMe su RunPod

1. Assicurati di avere tutte le dipendenze installate:
   ```
pip install -r requirements.txt
   ```
2. Avvia l'applicazione:
   ```
sh start.sh
   ```
3. L'app sarà accessibile via Gradio sull'host e porta specificati in config.yaml.

Per abilitare Neptune, modifica il flag NEPTUNE_ENABLED nel codice (disabilitato di default). 