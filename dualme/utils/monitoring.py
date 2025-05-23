import time
import logging
import psutil
import GPUtil
from datetime import datetime
from pathlib import Path
import json

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/monitoring.log'),
        logging.StreamHandler()
    ]
)

class PerformanceMonitor:
    def __init__(self):
        self.metrics_file = Path('/workspace/logs/performance_metrics.json')
        self.metrics = self._load_metrics()
        
    def _load_metrics(self):
        """Carica le metriche esistenti o crea un nuovo file"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {
            'total_requests': 0,
            'total_cost': 0.0,
            'average_times': {
                'model_loading': 0.0,
                'keypoint_mask_generation': 0.0,
                'densepose_inference': 0.0,
                'image_generation': 0.0
            },
            'costs': {
                'first_request': 0.014652,  # Costo prima richiesta
                'subsequent_request': 0.0048092  # Costo richieste successive
            }
        }
    
    def _save_metrics(self):
        """Salva le metriche nel file JSON"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def start_request(self):
        """Inizia il monitoraggio di una nuova richiesta"""
        return {
            'start_time': time.time(),
            'is_first_request': self.metrics['total_requests'] == 0
        }
    
    def end_request(self, request_data, phase_times):
        """Termina il monitoraggio di una richiesta e aggiorna le metriche"""
        end_time = time.time()
        total_time = end_time - request_data['start_time']
        
        # Aggiorna i tempi medi per fase
        for phase, time_taken in phase_times.items():
            current_avg = self.metrics['average_times'][phase]
            self.metrics['average_times'][phase] = (
                (current_avg * self.metrics['total_requests'] + time_taken) /
                (self.metrics['total_requests'] + 1)
            )
        
        # Calcola il costo
        cost = (self.metrics['costs']['first_request'] if request_data['is_first_request']
                else self.metrics['costs']['subsequent_request'])
        
        # Aggiorna le metriche totali
        self.metrics['total_requests'] += 1
        self.metrics['total_cost'] += cost
        
        # Salva le metriche
        self._save_metrics()
        
        # Log delle performance
        logging.info(f"Richiesta completata in {total_time:.2f} secondi")
        logging.info(f"Costo della richiesta: ${cost:.6f}")
        logging.info(f"Costo totale accumulato: ${self.metrics['total_cost']:.6f}")
        
        return {
            'total_time': total_time,
            'cost': cost,
            'phase_times': phase_times
        }
    
    def get_system_metrics(self):
        """Ottiene le metriche del sistema"""
        try:
            gpu = GPUtil.getGPUs()[0]  # Prendi la prima GPU
            return {
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_temperature': gpu.temperature,
                'cpu_percent': psutil.cpu_percent(),
                'ram_percent': psutil.virtual_memory().percent
            }
        except Exception as e:
            logging.error(f"Errore nel recupero delle metriche del sistema: {str(e)}")
            return None
    
    def log_system_metrics(self):
        """Registra le metriche del sistema"""
        metrics = self.get_system_metrics()
        if metrics:
            logging.info("Metriche del sistema:")
            logging.info(f"GPU Memory: {metrics['gpu_memory_used']}MB/{metrics['gpu_memory_total']}MB")
            logging.info(f"GPU Temperature: {metrics['gpu_temperature']}°C")
            logging.info(f"CPU Usage: {metrics['cpu_percent']}%")
            logging.info(f"RAM Usage: {metrics['ram_percent']}%")

def get_performance_monitor():
    """Factory function per ottenere l'istanza del monitor"""
    return PerformanceMonitor() 