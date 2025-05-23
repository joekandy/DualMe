import os
import subprocess
from pathlib import Path

def setup_backup_cron():
    """Configura il cron job per i backup giornalieri"""
    try:
        # Crea il comando cron per eseguire il backup ogni giorno alle 3 AM
        cron_command = "0 3 * * * cd /workspace && python -m dualme.utils.backup >> /workspace/logs/cron.log 2>&1"
        
        # Aggiungi il comando al crontab
        subprocess.run(['crontab', '-l'], capture_output=True)
        subprocess.run(['echo', cron_command, '|', 'crontab', '-'], shell=True)
        
        print("Backup automatico configurato con successo")
        print("Il backup verrà eseguito ogni giorno alle 3 AM")
        
    except Exception as e:
        print(f"Errore durante la configurazione del backup automatico: {str(e)}")

if __name__ == "__main__":
    setup_backup_cron() 