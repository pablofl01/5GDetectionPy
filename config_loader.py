#!/usr/bin/env python3
"""
Module to load and manage configuration from config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Class to manage application configuration."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initializes configuration from YAML file.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Loads YAML configuration file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a configuration value using dot notation.
        
        Args:
            key: Key in dot notation (e.g., 'rf.gscn')
            default: Default value if not exists
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    # === RF CONFIGURATION PROPERTIES ===
    
    @property
    def gscn(self) -> int:
        """GSCN (Global Synchronization Channel Number)."""
        return self.get('rf.gscn', 7929)
    
    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        return float(self.get('rf.sample_rate', 19.5e6))
    
    @property
    def gain(self) -> float:
        """Receiver gain in dB."""
        return float(self.get('rf.gain', 50))
    
    @property
    def scs(self) -> int:
        """Subcarrier spacing in kHz."""
        return self.get('rf.scs', 30)
    
    @property
    def antenna(self) -> str:
        """Antenna to use."""
        return self.get('rf.antenna', 'RX2')
    
    # === PROCESSING PROPERTIES ===
    
    @property
    def nrb_ssb(self) -> int:
        """Number of Resource Blocks for SSB."""
        return self.get('processing.nrb_ssb', 20)
    
    @property
    def nrb_demod(self) -> int:
        """Number of Resource Blocks for demodulation."""
        return self.get('processing.nrb_demod', 45)
    
    @property
    def n_symbols_display(self) -> int:
        """Number of OFDM symbols to demodulate."""
        return self.get('processing.n_symbols_display', 14)
    
    @property
    def search_bw(self) -> float:
        """Frequency search bandwidth in kHz."""
        return float(self.get('processing.search_bw', 90))
    
    @property
    def detection_threshold(self) -> float:
        """SSB detection threshold."""
        return float(self.get('processing.detection_threshold', 1e-3))
    
    # === MONITORING PROPERTIES ===
    
    @property
    def monitor_time(self) -> float:
        """Total monitoring time in seconds."""
        return float(self.get('monitoring.monitor_time', 0.57))
    
    @property
    def interval(self) -> float:
        """Interval between captures in seconds."""
        return float(self.get('monitoring.interval', 0.057))
    
    @property
    def frames_per_capture(self) -> int:
        """Number of 5G NR frames per capture."""
        return self.get('monitoring.frames_per_capture', 1)
    
    @property
    def save_captures(self) -> bool:
        """Save captures to disk."""
        return self.get('monitoring.save_captures', False)
    
    @property
    def captures_dir(self) -> str:
        """Directory to save captures."""
        return self.get('monitoring.captures_dir', 'capturas_disco')
    
    # === VISUALIZATION PROPERTIES ===
    
    @property
    def enable_gui(self) -> bool:
        """Show graphical interface."""
        return self.get('visualization.enable_gui', True)
    
    @property
    def figure_size(self) -> tuple:
        """Figure size (width, height) in inches."""
        size = self.get('visualization.figure_size', [12, 8])
        return tuple(size)
    
    @property
    def colormap(self) -> str:
        """Colormap for resource grid."""
        return self.get('visualization.colormap', 'jet')
    
    @property
    def interpolation(self) -> str:
        """Image interpolation."""
        return self.get('visualization.interpolation', 'nearest')
    
    @property
    def verbose(self) -> bool:
        """Display detailed information in console."""
        return self.get('visualization.verbose', False)
    
    # === EXPORT PROPERTIES ===
    
    @property
    def output_dir(self) -> str:
        """Export directory."""
        return self.get('export.output_dir', 'resultados')
    
    @property
    def save_plots(self) -> bool:
        """Save plots as PNG images."""
        return self.get('export.save_plots', False)
    
    @property
    def use_timestamp(self) -> bool:
        """Include timestamp in filenames."""
        return self.get('export.use_timestamp', True)
    
    # === DEVICE PROPERTIES ===
    
    @property
    def device_index(self) -> Optional[int]:
        """USRP device index."""
        return self.get('device.index')
    
    @property
    def device_serial(self) -> Optional[str]:
        """USRP device serial number."""
        return self.get('device.serial')
    
    @property
    def device_args(self) -> str:
        """Additional device arguments."""
        return self.get('device.args', '')
    
    def __repr__(self) -> str:
        """Configuration representation."""
        return f"Config(file='{self.config_file}', gscn={self.gscn}, scs={self.scs}kHz, sr={self.sample_rate/1e6:.1f}MHz)"


# Global configuration instance
_config_instance: Optional[Config] = None


def load_config(config_file: str = "config.yaml") -> Config:
    """
    Loads or returns the global configuration instance.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_file)
    
    return _config_instance


def get_config() -> Config:
    """
    Gets the global configuration instance.
    If it doesn't exist, loads it with the default file.
    
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config()
    
    return _config_instance
