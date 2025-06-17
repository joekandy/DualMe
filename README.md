# 🎯 DualMe - Virtual Try-On 2.0

**DualMe** è una soluzione avanzata di Virtual Try-On che utilizza tecniche di deep learning per creare esperienze di prova virtuale altamente realistiche e ad alta risoluzione (1024x768).

## 🌟 Caratteristiche Principali

- ✅ **Alta Risoluzione**: Output a 1024x768 per massimo realismo
- ✅ **Architettura Avanzata**: Algoritmo proprietario senza DensePose
- ✅ **Dataset VITON-HD**: Supporto nativo per training ad alta qualità
- ✅ **API Ready**: Interfaccia Gradio per facile integrazione
- ✅ **Cloud Deploy**: Ottimizzato per RunPod e servizi cloud

## 🚀 Quick Start

### 🌐 RunPod Deployment (Raccomandato)

1. **Crea un'istanza RunPod** con supporto GPU
2. **Clona il repository:**
```bash
git clone https://github.com/joekandy/DualMe.git
cd DualMe
```

3. **Configura i download dei modelli** in `setup_runpod.sh`:
   - Carica i file modello (`virtual_tryon.pth`, `virtual_tryon_dc.pth`) su Google Drive
   - Ottieni i link condivisibili e aggiorna lo script

4. **Esegui lo script di setup:**
```bash
chmod +x setup_runpod.sh
./setup_runpod.sh
```

5. **Avvia l'applicazione:**
```bash
./start.sh
```

### 💻 Installazione Locale

1. **Clona il repository:**
```bash
git clone https://github.com/joekandy/DualMe.git
cd DualMe
```

2. **Installa le dipendenze:**
```bash
pip install -r requirements.txt
```

3. **Scarica i file modello:**
   - Posiziona `virtual_tryon.pth` e `virtual_tryon_dc.pth` nella directory root

4. **Avvia l'applicazione:**
```bash
./start.sh
```

### 🌍 Accesso alle Applicazioni

Le applicazioni saranno disponibili su:
- **Fashion Virtual Try-On**: http://localhost:7860
- **DualMe App**: http://localhost:7861

## 🏗️ Struttura del Progetto

```
DualMe/
├── dualme/                    # Core DualMe application
│   ├── app/                   # Gradio applications
│   └── utils/                 # Utilities and helpers
├── Fashion Virtual Try On/    # Standalone VTO app
├── docs/                      # Documentation
├── archived_files/            # Archived components
├── requirements.txt           # Python dependencies
├── start.sh                   # Startup script
├── setup_runpod.sh           # RunPod setup script
├── Dockerfile                 # Docker configuration
└── README.md                  # This file
```

## ⚙️ Requisiti Tecnici

### Hardware Minimo
- **GPU**: NVIDIA con almeno 8GB VRAM
- **RAM**: 16GB
- **Storage**: 50GB liberi

### Software
- **Python**: 3.8+
- **CUDA**: 11.8+
- **PyTorch**: Con supporto CUDA

## 🎨 Come Usare

1. **Carica un'immagine della persona**
2. **Carica un'immagine del capo d'abbigliamento**
3. **Configura parametri (opzionale):**
   - Steps: numero di passi (default: 30)
   - Scale: intensità effetto (default: 2.5)
   - Seed: per riproducibilità (default: 42)
4. **Clicca "Genera" e attendi il risultato**

## 🔧 Troubleshooting

### Errori Comuni

**❌ CUDA non disponibile**
```bash
# Verifica installazione CUDA
nvcc --version
nvidia-smi
```

**❌ Memoria insufficiente**
- Riduci dimensione immagini
- Diminuisci parametro `steps`
- Chiudi altre applicazioni GPU

**❌ Modelli mancanti**
- Verifica presenza file `.pth` nella root
- Controlla permessi di lettura

### Supporto

Per problemi o domande:
1. Controlla la [documentazione](docs/)
2. Apri una [issue su GitHub](https://github.com/joekandy/DualMe/issues)

## 📄 Licenza

Questo progetto è proprietario e protetto da copyright. Basato su componenti open-source sotto licenza MIT.

## 🤝 Contributi

Attualmente questo è un progetto proprietario. Per collaborazioni commerciali, contattaci tramite GitHub.

---

**DualMe Virtual Try-On 2.0** - Sviluppato per la prossima generazione di esperienze di shopping online. 