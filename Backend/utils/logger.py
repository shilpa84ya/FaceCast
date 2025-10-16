"""
Logging System for FaceCast
Provides centralized logging with file output and console capture
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

class FaceCastLogger:
    """Centralized logging system for FaceCast application"""
    
    def __init__(self, log_dir: str = "logs"):
        # Get the directory where this script is located
        current_dir = Path(__file__).parent.parent  # Go up from utils to Backend
        self.log_dir = current_dir / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create different log files for different purposes
        self.log_files = {
            'app': self.log_dir / 'facecast_app.log',
            'security': self.log_dir / 'facecast_security.log',
            'errors': self.log_dir / 'facecast_errors.log',
            'access': self.log_dir / 'facecast_access.log'
        }
        
        # Initialize loggers
        self.loggers = {}
        self._setup_loggers()
        
    def _setup_loggers(self):
        """Setup different loggers for different purposes"""
        
        # Common formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        for log_type, log_file in self.log_files.items():
            logger = logging.getLogger(f'facecast.{log_type}')
            logger.setLevel(logging.INFO)
            
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Console handler for errors
            if log_type == 'errors':
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            self.loggers[log_type] = logger
    
    def log_info(self, message: str, log_type: str = 'app'):
        """Log info message"""
        if log_type in self.loggers:
            self.loggers[log_type].info(message)
        else:
            self.loggers['app'].info(f"[{log_type}] {message}")
    
    def log_error(self, message: str, log_type: str = 'errors'):
        """Log error message"""
        if log_type in self.loggers:
            self.loggers[log_type].error(message)
        else:
            self.loggers['errors'].error(f"[{log_type}] {message}")
    
    def log_warning(self, message: str, log_type: str = 'app'):
        """Log warning message"""
        if log_type in self.loggers:
            self.loggers[log_type].warning(message)
        else:
            self.loggers['app'].warning(f"[{log_type}] {message}")
    
    def log_security_event(self, event_data: Dict[str, Any]):
        """Log security events with structured data"""
        event_data['timestamp'] = datetime.now().isoformat()
        message = json.dumps(event_data)
        self.loggers['security'].info(message)
    
    def log_access(self, ip: str, method: str, endpoint: str, status: int, duration: float):
        """Log API access"""
        message = f"{ip} - {method} {endpoint} - {status} - {duration:.2f}ms"
        self.loggers['access'].info(message)
    
    def get_log_files_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about log files"""
        info = {}
        for log_type, log_file in self.log_files.items():
            if log_file.exists():
                stat = log_file.stat()
                info[log_file.name] = {
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'lines': self._count_lines(log_file)
                }
        return info
    
    def get_log_files(self):
        """Get list of available log files (compatibility method)"""
        try:
            files = [f.name for f in self.log_dir.glob("*.log")]
            return sorted(files, reverse=True)
        except Exception:
            return []
    
    def get_log_stats(self):
        """Get log statistics (compatibility method)"""
        try:
            files = self.get_log_files()
            total_size = 0
            for file in files:
                filepath = self.log_dir / file
                if filepath.exists():
                    total_size += filepath.stat().st_size
            
            return {
                'total_files': len(files),
                'latest_file': files[0] if files else None,
                'total_size': total_size
            }
        except Exception:
            return {'total_files': 0, 'latest_file': None, 'total_size': 0}
    
    def read_log_file(self, filename, lines=1000):
        """Read log file content (compatibility method)"""
        try:
            filepath = self.log_dir / filename
            if not filepath.exists():
                return []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.readlines()
            
            return content[-lines:] if lines > 0 else content
        except Exception:
            return []
    
    def error(self, message, exception=None, category='errors'):
        """Error method for compatibility"""
        if exception:
            message = f"{message}: {str(exception)}"
        self.log_error(message, category)
    
    def info(self, message, category='app'):
        """Info method for compatibility"""
        self.log_info(message, category)
    
    def warning(self, message, category='app'):
        """Warning method for compatibility"""
        self.log_warning(message, category)
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def cleanup_old_logs(self, days: int = 30):
        """Clean up log files older than specified days"""
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.log_info(f"Deleted old log file: {log_file.name}")
                except Exception as e:
                    self.log_error(f"Failed to delete log file {log_file.name}: {e}")

# Global logger instance
_logger_instance: Optional[FaceCastLogger] = None

def initialize_logger(log_dir: str = "logs") -> FaceCastLogger:
    """Initialize the global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = FaceCastLogger(log_dir)
    return _logger_instance

def get_logger() -> FaceCastLogger:
    """Get the global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = initialize_logger()
    return _logger_instance

# Convenience functions for backward compatibility
def log_info(message: str, log_type: str = 'app'):
    """Log info message"""
    get_logger().log_info(message, log_type)

def log_error(message: str, log_type: str = 'errors'):
    """Log error message"""
    get_logger().log_error(message, log_type)

def log_warning(message: str, log_type: str = 'app'):
    """Log warning message"""
    get_logger().log_warning(message, log_type)

def log_security_event(event_type_or_data, details=None, ip_address=None):
    """Log security events - supports both old and new format"""
    if isinstance(event_type_or_data, dict):
        # New format: single dictionary
        get_logger().log_security_event(event_type_or_data)
    else:
        # Old format: separate parameters
        event_data = {
            'event_type': event_type_or_data,
            'details': details or {},
            'ip_address': ip_address or 'unknown'
        }
        get_logger().log_security_event(event_data)

def log_access(ip: str, method: str, endpoint: str, status: int, duration: float):
    """Log API access"""
    get_logger().log_access(ip, method, endpoint, status, duration)

# Create sample log files on import
def create_sample_logs():
    """Create sample log files for testing"""
    logger = get_logger()
    
    # Sample application logs
    logger.log_info("FaceCast application initialized")
    logger.log_info("Database connection established")
    logger.log_info("Face recognition system loaded")
    
    # Sample security logs
    logger.log_security_event({
        "event_type": "SYSTEM_START",
        "details": {"component": "FaceCast", "version": "1.0"}
    })
    
    # Sample access logs
    logger.log_access("127.0.0.1", "GET", "/api/elections", 200, 45.2)
    logger.log_access("127.0.0.1", "POST", "/api/login", 200, 150.8)

# Initialize on import (commented out to avoid import-time errors)
# if __name__ != "__main__":
#     create_sample_logs()