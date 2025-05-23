import os
import shutil
import datetime
import logging
from pathlib import Path

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/backup.log'),
        logging.StreamHandler()
    ]
)

class BackupManager:
    def __init__(self, workspace_dir="/workspace", backup_dir="/workspace/backups"):
        self.workspace_dir = Path(workspace_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def create_backup(self):
        """Crea un backup dei dati importanti"""
        try:
            # Crea il nome del backup con timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            # Crea la directory del backup
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup dei modelli
            models_dir = self.workspace_dir / "models"
            if models_dir.exists():
                shutil.copytree(models_dir, backup_path / "models")
                logging.info(f"Backup dei modelli completato in {backup_path / 'models'}")
            
            # Backup dei checkpoint
            checkpoints_dir = self.workspace_dir / "checkpoints"
            if checkpoints_dir.exists():
                shutil.copytree(checkpoints_dir, backup_path / "checkpoints")
                logging.info(f"Backup dei checkpoint completato in {backup_path / 'checkpoints'}")
            
            # Backup dei log
            logs_dir = self.workspace_dir / "logs"
            if logs_dir.exists():
                shutil.copytree(logs_dir, backup_path / "logs")
                logging.info(f"Backup dei log completato in {backup_path / 'logs'}")
            
            # Rimuovi i backup vecchi (mantieni solo gli ultimi 7 giorni)
            self._cleanup_old_backups()
            
            return True
            
        except Exception as e:
            logging.error(f"Errore durante il backup: {str(e)}")
            return False
    
    def _cleanup_old_backups(self, days=7):
        """Rimuove i backup più vecchi di 'days' giorni"""
        try:
            current_time = datetime.datetime.now()
            for backup in self.backup_dir.glob("backup_*"):
                backup_time = datetime.datetime.strptime(
                    backup.name.split("_")[1], 
                    "%Y%m%d_%H%M%S"
                )
                if (current_time - backup_time).days > days:
                    shutil.rmtree(backup)
                    logging.info(f"Rimosso backup vecchio: {backup}")
        except Exception as e:
            logging.error(f"Errore durante la pulizia dei backup: {str(e)}")

def run_backup():
    """Funzione principale per eseguire il backup"""
    backup_manager = BackupManager()
    success = backup_manager.create_backup()
    if success:
        logging.info("Backup completato con successo")
    else:
        logging.error("Backup fallito")

if __name__ == "__main__":
    run_backup() 