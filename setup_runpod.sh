#!/bin/bash
# Script di setup completo per DualMe VTO su RunPod

echo "🚀 SETUP DUALME VTO SU RUNPOD"
echo "============================="

# Aggiorna il sistema
echo "📦 Aggiornamento sistema..."
apt-get update -y

# Installa dipendenze di sistema se necessarie
echo "🔧 Installazione dipendenze sistema..."
apt-get install -y wget curl git

# Installa dipendenze Python
echo "🐍 Installazione dipendenze Python..."
pip install --upgrade pip
pip install -r requirements.txt

# Download modelli da Google Drive o altro servizio
echo "📥 Download modelli..."
echo "IMPORTANTE: Inserisci i link dei tuoi modelli nelle variabili sottostanti"

# SOSTITUISCI QUESTI LINK CON I TUOI LINK EFFETTIVI
MODEL_1_URL="YOUR_GOOGLE_DRIVE_LINK_virtual_tryon.pth"
MODEL_2_URL="YOUR_GOOGLE_DRIVE_LINK_virtual_tryon_dc.pth"

# Installa gdown per Google Drive
pip install gdown

# Download modelli (sostituisci con i tuoi link)
if [ "$MODEL_1_URL" != "YOUR_GOOGLE_DRIVE_LINK_virtual_tryon.pth" ]; then
    echo "📥 Scaricando virtual_tryon.pth..."
    gdown "$MODEL_1_URL" -O virtual_tryon.pth
else
    echo "⚠️  CONFIGURA I LINK DEI MODELLI NEL SCRIPT!"
fi

if [ "$MODEL_2_URL" != "YOUR_GOOGLE_DRIVE_LINK_virtual_tryon_dc.pth" ]; then
    echo "📥 Scaricando virtual_tryon_dc.pth..."
    gdown "$MODEL_2_URL" -O virtual_tryon_dc.pth
else
    echo "⚠️  CONFIGURA I LINK DEI MODELLI NEL SCRIPT!"
fi

# Verifica che i modelli siano stati scaricati
echo "🔍 Verifica modelli..."
if [ -f "virtual_tryon.pth" ] && [ -f "virtual_tryon_dc.pth" ]; then
    echo "✅ Modelli scaricati correttamente:"
    ls -lh *.pth
else
    echo "❌ Modelli mancanti! Controlla i link e riprova."
    exit 1
fi

# Rendi eseguibile lo script di avvio
chmod +x start.sh

echo "🎉 Setup completato!"
echo "📋 Per avviare l'applicazione, esegui:"
echo "    ./start.sh"
echo ""
echo "🌐 L'applicazione sarà disponibile su:"
echo "    http://localhost:7860 (Gradio Fashion Virtual Try On)"
echo "    http://localhost:7861 (DualMe App)" 