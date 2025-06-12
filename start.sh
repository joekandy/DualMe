#!/bin/bash

# Crea le directory necessarie
mkdir -p /workspace/logs
mkdir -p /workspace/backups
mkdir -p /workspace/models
mkdir -p /workspace/checkpoints

# Configura il backup automatico
python -m dualme.utils.schedule_backup

# Avvia il servizio cron
service cron start

# Verifica se i modelli sono già presenti
if [ ! -f "/workspace/models/upper_model.pth" ] || [ ! -f "/workspace/models/dressed_model.pth" ]; then
    echo "Configurazione dei modelli..."
    python -m dualme.utils.setup_models
fi

# Verifica lo spazio su disco
DISK_USAGE=$(df -h /workspace | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "WARNING: Disk usage is above 90%"
fi

# Verifica la memoria GPU
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$GPU_MEMORY" -gt 14000 ]; then
    echo "WARNING: GPU memory usage is high"
fi

# Avvia l'applicazione con gestione degli errori
while true; do
    echo "Starting application..."
    python -m dualme.app.main
    
    # Se l'applicazione si chiude, aspetta 5 secondi prima di riavviare
    echo "Application stopped. Restarting in 5 seconds..."
    sleep 5
done

# Avvio DualMe su RunPod
python3 -m dualme.app.gradio_app 