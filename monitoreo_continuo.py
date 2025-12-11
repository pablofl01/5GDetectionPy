#!/usr/bin/env python3
"""
Script de Monitoreo Continuo 5G NR usando USRP B210
Equivalente Python del script MonitoreoContinuoFunciones.m
Requiere: uhd, numpy, scipy, matplotlib
"""

import uhd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import signal
from scipy.fft import fft, ifft, fftshift
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import warnings
import argparse
import yaml
import os
from pathlib import Path

warnings.filterwarnings('ignore')


def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    """
    Carga configuración desde archivo YAML
    
    Args:
        config_file: Ruta al archivo de configuración
        
    Returns:
        Diccionario con la configuración
    """
    if not os.path.exists(config_file):
        print(f'⚠ Archivo de configuración no encontrado: {config_file}')
        print('  Usando valores por defecto')
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f'✓ Configuración cargada desde: {config_file}')
        return config if config else {}
    except Exception as e:
        print(f'❌ Error al cargar configuración: {e}')
        print('  Usando valores por defecto')
        return {}


def merge_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Combina configuración de archivo con argumentos de línea de comandos
    Los argumentos de línea de comandos tienen prioridad
    
    Args:
        config: Configuración desde archivo
        args: Argumentos de línea de comandos
        
    Returns:
        Configuración fusionada
    """
    # Crear estructura por defecto
    merged = {
        'device': config.get('device', {}),
        'rf': config.get('rf', {}),
        'processing': config.get('processing', {}),
        'monitoring': config.get('monitoring', {}),
        'visualization': config.get('visualization', {}),
        'simulation': config.get('simulation', {}),
        'export': config.get('export', {})
    }
    
    # Los argumentos de CLI tienen prioridad sobre el archivo
    # Solo sobrescribir si el argumento fue explícitamente proporcionado
    
    # Device
    if hasattr(args, 'device_index') and args.device_index is not None:
        merged['device']['index'] = args.device_index
    if hasattr(args, 'device_serial') and args.device_serial is not None:
        merged['device']['serial'] = args.device_serial
    if hasattr(args, 'device_args') and args.device_args:
        merged['device']['args'] = args.device_args
    
    # RF
    if hasattr(args, 'gscn') and args.gscn is not None:
        merged['rf']['gscn'] = args.gscn
    if hasattr(args, 'sample_rate') and args.sample_rate is not None:
        merged['rf']['sample_rate'] = args.sample_rate
    if hasattr(args, 'gain') and args.gain is not None:
        merged['rf']['gain'] = args.gain
    if hasattr(args, 'scs') and args.scs is not None:
        merged['rf']['scs'] = args.scs
    
    # Processing
    if hasattr(args, 'n_symbols_display') and args.n_symbols_display is not None:
        merged['processing']['n_symbols_display'] = args.n_symbols_display
    
    # Monitoring
    if hasattr(args, 'monitor_time') and args.monitor_time is not None:
        merged['monitoring']['monitor_time'] = args.monitor_time
    if hasattr(args, 'interval') and args.interval is not None:
        merged['monitoring']['interval'] = args.interval
    if hasattr(args, 'frames') and args.frames is not None:
        merged['monitoring']['frames_per_capture'] = args.frames
    
    # Visualization
    if hasattr(args, 'no_gui') and args.no_gui:
        merged['visualization']['enable_gui'] = False
    if hasattr(args, 'verbose') and args.verbose:
        merged['visualization']['verbose'] = True
    
    # Simulation
    if hasattr(args, 'simulate') and args.simulate:
        merged['simulation']['enabled'] = True
    
    return merged


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Obtiene valor de configuración usando notación de punto
    
    Args:
        config: Diccionario de configuración
        path: Ruta al valor (ej: 'rf.gscn')
        default: Valor por defecto si no existe
        
    Returns:
        Valor de configuración o default
    """
    keys = path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value if value is not None else default


def list_usrp_devices():
    """
    Lista todos los dispositivos USRP disponibles
    
    Returns:
        Lista de diccionarios con información de cada dispositivo
    """
    print('\n=== DISPOSITIVOS USRP DISPONIBLES ===')
    device_addrs = uhd.find("")
    
    if not device_addrs:
        print('No se encontraron dispositivos USRP conectados.')
        return []
    
    devices = []
    for idx, addr in enumerate(device_addrs):
        print(f'\n[{idx}] Dispositivo encontrado:')
        device_info = {}
        for key in addr.keys():
            value = addr.get(key)
            print(f'    {key}: {value}')
            device_info[key] = value
        devices.append(device_info)
    
    print('\n' + '='*40)
    return devices


def select_usrp_device(device_index: Optional[int] = None, 
                       device_serial: Optional[str] = None,
                       device_args: str = "") -> str:
    """
    Selecciona un dispositivo USRP específico
    
    Args:
        device_index: Índice del dispositivo (0, 1, 2, ...)
        device_serial: Número de serie del dispositivo
        device_args: Argumentos adicionales del dispositivo
        
    Returns:
        String de argumentos del dispositivo para MultiUSRP
    """
    devices = list_usrp_devices()
    
    if not devices:
        raise RuntimeError("No hay dispositivos USRP disponibles")
    
    # Si se especificó índice
    if device_index is not None:
        if 0 <= device_index < len(devices):
            selected = devices[device_index]
            print(f'\n✓ Seleccionado dispositivo [{device_index}]: {selected.get("serial", "N/A")}')
            # Construir device_args con el serial
            if 'serial' in selected:
                return f"serial={selected['serial']},{device_args}" if device_args else f"serial={selected['serial']}"
            return device_args
        else:
            raise ValueError(f"Índice {device_index} fuera de rango. Hay {len(devices)} dispositivos.")
    
    # Si se especificó serial
    if device_serial is not None:
        for dev in devices:
            if dev.get('serial') == device_serial:
                print(f'\n✓ Seleccionado dispositivo con serial: {device_serial}')
                return f"serial={device_serial},{device_args}" if device_args else f"serial={device_serial}"
        raise ValueError(f"No se encontró dispositivo con serial: {device_serial}")
    
    # Si solo hay uno, usarlo automáticamente
    if len(devices) == 1:
        selected = devices[0]
        print(f'\n✓ Usando único dispositivo disponible: {selected.get("serial", "N/A")}')
        if 'serial' in selected:
            return f"serial={selected['serial']},{device_args}" if device_args else f"serial={selected['serial']}"
        return device_args
    
    # Si hay múltiples y no se especificó, pedir selección interactiva
    print(f'\n⚠ Hay {len(devices)} dispositivos. Especifica --device-index o --device-serial')
    print('   o usa --list-devices para ver la lista completa.')
    raise RuntimeError("Múltiples dispositivos encontrados. Especifica cuál usar.")


class USRPB210Receiver:
    """Clase para manejar el USRP B210"""
    
    def __init__(self, center_freq: float, sample_rate: float, gain: float, 
                 channels=[0], device_args=""):
        """
        Inicializa el receptor USRP B210
        
        Args:
            center_freq: Frecuencia central en Hz
            sample_rate: Tasa de muestreo en Hz
            gain: Ganancia del receptor en dB
            channels: Lista de canales a utilizar
            device_args: Argumentos adicionales para el dispositivo
        """
        self.usrp = uhd.usrp.MultiUSRP(device_args)
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.channels = channels
        
        # Configurar USRP
        self._configure()
        
    def _configure(self):
        """Configura los parámetros del USRP"""
        # Configurar tasa de muestreo
        self.usrp.set_rx_rate(self.sample_rate, 0)
        actual_rate = self.usrp.get_rx_rate(0)
        print(f"Tasa de muestreo solicitada: {self.sample_rate/1e6:.2f} MHz")
        print(f"Tasa de muestreo obtenida: {actual_rate/1e6:.2f} MHz")
        
        # Configurar frecuencia central
        tune_request = uhd.types.TuneRequest(self.center_freq)
        self.usrp.set_rx_freq(tune_request, 0)
        actual_freq = self.usrp.get_rx_freq(0)
        print(f"Frecuencia central: {actual_freq/1e6:.2f} MHz")
        
        # Configurar ganancia
        self.usrp.set_rx_gain(self.gain, 0)
        actual_gain = self.usrp.get_rx_gain(0)
        print(f"Ganancia: {actual_gain:.1f} dB")
        
        # Configurar antena
        self.usrp.set_rx_antenna("RX2", 0)
        print(f"Antena: {self.usrp.get_rx_antenna(0)}")
        
    def capture(self, duration: float) -> np.ndarray:
        """
        Captura una señal durante un tiempo específico
        
        Args:
            duration: Duración de la captura en segundos
            
        Returns:
            Array complejo con las muestras IQ capturadas
        """
        num_samples = int(duration * self.sample_rate)
        samples = np.zeros(num_samples, dtype=np.complex64)
        
        # Configurar stream
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = self.channels
        rx_streamer = self.usrp.get_rx_stream(stream_args)
        
        # Preparar buffer
        recv_buffer = np.zeros((1, 10000), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        
        # Comando de streaming
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = num_samples
        stream_cmd.stream_now = True
        rx_streamer.issue_stream_cmd(stream_cmd)
        
        # Recibir muestras
        samples_received = 0
        while samples_received < num_samples:
            num_rx_samps = rx_streamer.recv(recv_buffer, metadata, 1.0)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                print(f"Error en recepción: {metadata.strerror()}")
                break
                
            end_idx = min(samples_received + num_rx_samps, num_samples)
            samples[samples_received:end_idx] = recv_buffer[0, :end_idx-samples_received]
            samples_received = end_idx
            
        return samples
    
    def release(self):
        """Libera recursos del USRP"""
        self.usrp = None


class NR5GProcessor:
    """Clase para procesar señales 5G NR"""
    
    def __init__(self, scs_khz: int = 30, sample_rate: float = 19.5e6):
        """
        Inicializa el procesador 5G NR
        
        Args:
            scs_khz: Subcarrier spacing en kHz (15, 30, 60, 120, 240)
            sample_rate: Tasa de muestreo en Hz
        """
        self.scs_khz = scs_khz
        self.sample_rate = sample_rate
        self.nrb_ssb = 20  # 20 RBs para SSB
        self.n_subcarriers = self.nrb_ssb * 12
        self.fft_size = 2048  # Tamaño FFT para demodulación OFDM
        
    @staticmethod
    def gscn_to_frequency(gscn: int) -> float:
        """
        Convierte GSCN a frecuencia en Hz
        
        Args:
            gscn: Global Synchronization Channel Number
            
        Returns:
            Frecuencia en Hz
        """
        # Para FR1 (banda n78 típicamente)
        if 7499 <= gscn <= 22255:
            N = gscn - 7499
            freq_hz = 3000e6 + N * 1.44e6
            return freq_hz
        else:
            raise ValueError(f"GSCN {gscn} fuera de rango FR1")
    
    def frequency_correction(self, waveform: np.ndarray, 
                            search_bw_khz: float = 90) -> Tuple[np.ndarray, float, int]:
        """
        Corrección de frecuencia buscando PSS
        
        Args:
            waveform: Señal IQ recibida
            search_bw_khz: Ancho de banda de búsqueda en kHz
            
        Returns:
            waveform_corrected: Señal corregida en frecuencia
            freq_offset: Offset de frecuencia detectado en Hz
            nid2: NID2 detectado (0, 1, o 2)
        """
        # Generar secuencias PSS para NID2 = 0, 1, 2
        pss_sequences = [self._generate_pss(nid2) for nid2 in range(3)]
        
        # Rango de búsqueda de frecuencia (pasos de medio subcarrier como MATLAB)
        scs_hz = self.scs_khz * 1000
        search_range = search_bw_khz * 1000
        # Pasos de medio subcarrier
        freq_offsets = np.arange(-search_range, search_range + scs_hz/2, scs_hz/2)
        
        max_corr = 0
        best_offset = 0
        best_nid2 = 0
        
        # FFT size pequeño para búsqueda inicial (como en MATLAB: syncNfft = 256)
        fft_size_search = 256
        
        # Correlación cruzada en diferentes offsets
        for freq_off in freq_offsets:
            # Aplicar offset de frecuencia
            t = np.arange(len(waveform)) / self.sample_rate
            corrected = waveform * np.exp(-1j * 2 * np.pi * freq_off * t)
            
            # Demodular OFDM con FFT pequeño para búsqueda
            cp_samples = 18 if self.scs_khz == 30 else 20
            symbol_length = fft_size_search + cp_samples
            
            # Solo procesar primer símbolo OFDM para búsqueda rápida
            if len(corrected) < symbol_length:
                continue
            
            # Demodular símbolos 0 y 1 (PSS está en símbolo 1)
            grid_search = np.zeros((fft_size_search, 2), dtype=np.complex128)
            for sym_idx in range(2):
                start = sym_idx * symbol_length + cp_samples
                end = start + fft_size_search
                if end <= len(corrected):
                    symbol = corrected[start:end]
                    freq_domain = fft(symbol)
                    grid_search[:, sym_idx] = fftshift(freq_domain)
            
            # PSS está en el símbolo 1 (índice 1)
            pss_symbol = grid_search[:, 1]
            
            # Correlación con cada PSS
            for nid2, pss_ref in enumerate(pss_sequences):
                # Extraer subportadoras centrales (127 para PSS)
                center = len(pss_symbol) // 2
                half_len = len(pss_ref) // 2
                if center + half_len <= len(pss_symbol):
                    pss_rx = pss_symbol[center-half_len:center+half_len+1][:len(pss_ref)]
                    
                    if len(pss_rx) == len(pss_ref):
                        corr = np.abs(np.sum(pss_rx * np.conj(pss_ref)))**2
                        
                        if corr > max_corr:
                            max_corr = corr
                            best_offset = freq_off
                            best_nid2 = nid2
        
        # Aplicar mejor offset (coarse frequency correction)
        t = np.arange(len(waveform)) / self.sample_rate
        waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * best_offset * t)
        
        # Fine frequency correction usando CP (simplificado)
        # En MATLAB se hace con hSSBurstFineFrequencyOffset
        fine_offset = self._fine_frequency_offset(waveform_corrected)
        total_offset = best_offset + fine_offset
        
        waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * total_offset * t)
        
        return waveform_corrected, total_offset, best_nid2
    
    def _generate_pss(self, nid2: int) -> np.ndarray:
        """
        Genera secuencia PSS para NID2 dado
        
        Args:
            nid2: Physical layer cell identity group (0, 1, 2)
            
        Returns:
            Secuencia PSS de 127 elementos
        """
        # Secuencia m-sequence
        x = np.zeros(127, dtype=np.complex128)
        
        # Inicialización según 3GPP TS 38.211
        for n in range(127):
            m = (n + 43 * nid2) % 127
            x[n] = np.exp(-1j * np.pi * m * (m + 1) / 63)
            
        return x
    
    def _generate_sss(self, ncellid: int) -> np.ndarray:
        """
        Genera secuencia SSS para cell ID dado
        
        Args:
            ncellid: Physical Cell ID (0-1007)
            
        Returns:
            Secuencia SSS de 127 elementos
        """
        nid1 = ncellid // 3
        nid2 = ncellid % 3
        
        # Simplificación de generación SSS
        x0 = np.zeros(127, dtype=np.complex128)
        x1 = np.zeros(127, dtype=np.complex128)
        
        for n in range(127):
            m0 = (n + nid1) % 127
            m1 = (n + nid2) % 127
            x0[n] = np.exp(-1j * np.pi * m0 * (m0 + 1) / 63)
            x1[n] = np.exp(-1j * np.pi * m1 * (m1 + 1) / 63)
        
        return x0 * x1
    
    def _fine_frequency_offset(self, waveform: np.ndarray) -> float:
        """
        Corrección fina de frecuencia usando correlación del Cyclic Prefix
        Similar a hSSBurstFineFrequencyOffset en MATLAB
        
        Args:
            waveform: Señal IQ corregida en frecuencia (coarse)
            
        Returns:
            Fine frequency offset en Hz
        """
        # Parámetros OFDM
        fft_size = 256
        cp_samples = 18 if self.scs_khz == 30 else 20
        symbol_length = fft_size + cp_samples
        
        # Necesitamos al menos 4 símbolos OFDM (tamaño SSB)
        if len(waveform) < 4 * symbol_length:
            return 0.0
        
        # Multiplicar waveform por sí misma retrasada y conjugada
        delayed = np.concatenate([np.zeros(fft_size), waveform[:-fft_size]])
        cp_product = waveform * np.conj(delayed)
        
        # Filtro de suma móvil con ventana = CP length
        cp_xcorr = np.convolve(cp_product, np.ones(cp_samples), mode='same')
        
        # Suma móvil sobre 4 símbolos OFDM
        y = cp_xcorr.copy()
        for k in range(1, 4):
            delayed_xcorr = np.concatenate([np.zeros(k * symbol_length), 
                                           cp_xcorr[:-(k * symbol_length)]])
            y += delayed_xcorr
        
        # Extraer pico de correlación
        cp_corr_index = fft_size + cp_samples + 3 * symbol_length
        if cp_corr_index < len(y):
            scs_hz = self.scs_khz * 1000
            freq_offset = scs_hz * np.angle(y[cp_corr_index]) / (2 * np.pi)
            return freq_offset
        
        return 0.0
    
    def _estimate_timing_offset(self, waveform: np.ndarray, pss_ref: np.ndarray) -> Tuple[int, float]:
        """
        Estima el timing offset usando correlación con PSS
        
        Args:
            waveform: Señal IQ
            pss_ref: Secuencia PSS de referencia
            
        Returns:
            (timing_offset, correlation_peak): Offset en muestras y valor de correlación
        """
        # Generar versión temporal del PSS
        fft_size = 256
        pss_freq = np.zeros(fft_size, dtype=np.complex128)
        center = fft_size // 2
        half_len = len(pss_ref) // 2
        pss_freq[center-half_len:center+half_len+1][:len(pss_ref)] = pss_ref
        pss_time = ifft(fftshift(pss_freq))
        
        # Correlación con la señal
        search_length = min(len(waveform), 10000)  # Buscar en primeros 10000 samples
        correlation = np.correlate(waveform[:search_length], pss_time, mode='valid')
        
        # Encontrar pico de correlación
        peak_idx = np.argmax(np.abs(correlation))
        peak_value = np.abs(correlation[peak_idx])
        
        return peak_idx, peak_value
    
    def _ofdm_demodulate_simple(self, waveform: np.ndarray, 
                                n_symbols: int = 4) -> Optional[np.ndarray]:
        """
        Demodulación OFDM simplificada para SSB
        
        Args:
            waveform: Señal temporal
            n_symbols: Número de símbolos OFDM a demodular
            
        Returns:
            Grid de recursos [subcarriers x symbols]
        """
        # Parámetros OFDM para 5G NR SSB (SCS=30kHz)
        # FFT size para 20 RBs = 240 subcarriers -> usar potencia de 2 cercana
        fft_size = 256  # Potencia de 2 más cercana a 240
        cp_samples = 18 if self.scs_khz == 30 else 20  # CP para SCS=30kHz
        symbol_length = fft_size + cp_samples
        
        n_samples_needed = n_symbols * symbol_length
        if len(waveform) < n_samples_needed:
            return None
        
        grid = np.zeros((fft_size, n_symbols), dtype=np.complex128)
        
        for sym_idx in range(n_symbols):
            start = sym_idx * symbol_length + cp_samples
            end = start + fft_size
            
            if end > len(waveform):
                break
                
            symbol = waveform[start:end]
            freq_domain = fft(symbol)
            grid[:, sym_idx] = fftshift(freq_domain)
        
        return grid
    
    def ofdm_demodulate(self, waveform: np.ndarray, 
                       nrb: int = 45, n_symbols: int = 14) -> Optional[np.ndarray]:
        """
        Demodulación OFDM completa para resource grid
        
        Args:
            waveform: Señal temporal
            nrb: Número de resource blocks
            n_symbols: Número de símbolos OFDM a demodular
            
        Returns:
            Resource grid completo
        """
        
        # Parámetros OFDM según número de RBs
        n_subcarriers = nrb * 12
        # FFT size: siguiente potencia de 2 mayor que n_subcarriers
        fft_size = int(2**np.ceil(np.log2(n_subcarriers * 1.2)))  # Con margen
        
        cp_samples = int(fft_size * 0.07)  # ~7% CP nominal
        symbol_length = fft_size + cp_samples
        
        n_samples_needed = n_symbols * symbol_length
        if len(waveform) < n_samples_needed:
            # Ajustar número de símbolos si no hay suficientes muestras
            n_symbols = min(n_symbols, len(waveform) // symbol_length)
            if n_symbols == 0:
                return None
        
        grid = np.zeros((fft_size, n_symbols), dtype=np.complex128)
        
        for sym_idx in range(n_symbols):
            start = sym_idx * symbol_length + cp_samples
            end = start + fft_size
            
            if end > len(waveform):
                break
                
            symbol = waveform[start:end]
            freq_domain = fft(symbol)
            grid[:, sym_idx] = fftshift(freq_domain)
        
        # Extraer subportadoras centrales
        center = fft_size // 2
        half = n_subcarriers // 2
        grid_extracted = grid[center-half:center+half, :]
        
        return grid_extracted
    
    def find_ssb(self, waveform: np.ndarray, verbose: bool = False, n_symbols_display: int = 14) -> Tuple[bool, np.ndarray, int, int, dict]:
        """
        Detecta SSB en la señal
        
        Args:
            waveform: Señal IQ capturada
            verbose: Mostrar información de diagnóstico
            n_symbols_display: Número de símbolos OFDM a demodular y mostrar
            
        Returns:
            detected: Si se detectó SSB
            resource_grid: Grid de recursos
            ncellid: Physical Cell ID
            strongest_ssb_idx: Índice del SSB más fuerte (0-7)
            ssb_info: Diccionario con información del SSB
        """
        if verbose:
            print(f"  Muestras recibidas: {len(waveform)}, Potencia: {10*np.log10(np.mean(np.abs(waveform)**2)):.1f} dB")
        
        # Corrección de frecuencia
        corrected_waveform, freq_offset, nid2 = self.frequency_correction(waveform)
        
        if verbose:
            print(f"  Freq offset: {freq_offset:.1f} Hz, NID2: {nid2}")
        
        # Calcular timing offset usando PSS
        pss_ref = self._generate_pss(nid2)
        pss_position, pss_corr = self._estimate_timing_offset(corrected_waveform, pss_ref)
        
        if verbose:
            print(f"  Posición PSS: {pss_position} muestras (corr={pss_corr:.2e})")
        
        # El PSS está en el símbolo 1 del SSB (4 símbolos: PBCH, PSS, PBCH, SSS)
        # Para que SSB quede en símbolos 2-5 de la visualización:
        # - Símbolo 2 debe empezar donde empieza el SSB (símbolo 0 del SSB)
        # - PSS está en símbolo 1 del SSB
        # Necesitamos retroceder 1 símbolo OFDM desde la posición del PSS
        
        fft_size = 256
        cp_samples = 18 if self.scs_khz == 30 else 20
        symbol_length = fft_size + cp_samples
        
        # Calcular offset para que SSB empiece exactamente en símbolo 2 (índice 1)
        # Desde posición PSS (símbolo 1 del SSB) retrocedemos 1 símbolo
        # Luego avanzamos 1 símbolo para posicionar SSB en símbolo 2 de la visualización
        # Net: pss_position (que ya apunta al inicio del PSS)
        ssb_start_in_signal = pss_position - symbol_length  # Inicio del SSB (símbolo 0)
        target_symbol_offset = 1 * symbol_length  # Queremos SSB en símbolo 2 -> offset de 1 símbolo
        
        # El offset a aplicar es: llevamos el inicio del SSB al símbolo 1 de la señal
        timing_offset = ssb_start_in_signal - target_symbol_offset
        
        # Aplicar timing offset
        if timing_offset > 0 and timing_offset < len(corrected_waveform) - 10*symbol_length:
            corrected_waveform = corrected_waveform[timing_offset:]
        else:
            # Si el offset es negativo o muy grande, usar posición del PSS directamente
            if pss_position > symbol_length and pss_position < len(corrected_waveform) - 10*symbol_length:
                timing_offset = pss_position - 2*symbol_length  # Aproximación conservadora
                corrected_waveform = corrected_waveform[max(0, timing_offset):]
        
        if verbose:
            print(f"  Timing offset aplicado: {timing_offset} muestras")
            print(f"  SSB debería aparecer en símbolos 2-5 de la visualización")
        
        # Demodulación OFDM del grid completo (45 RBs, símbolos configurables)
        grid_full = self.ofdm_demodulate(corrected_waveform, nrb=45, n_symbols=n_symbols_display)
        
        if grid_full is None or grid_full.shape[1] < 6:
            if verbose:
                print("  ❌ No se pudo demodular suficientes símbolos OFDM")
            return False, None, 0, 0, {}
        
        # Extraer símbolos SSB del grid (símbolos 2-5 = índices 1:5 en Python)
        if grid_full.shape[1] < 6:
            if verbose:
                print(f'  ✗ Grid insuficiente: {grid_full.shape[1]} símbolos')
            return False, None, 0, 0, {}
        
        grid_ssb = grid_full[:, 1:5]  # Símbolos 2,3,4,5 (índices 1,2,3,4)
        
        # Detección de NID1 usando SSS
        max_corr = 0
        best_nid1 = 0
        
        if grid_ssb.shape[1] >= 4:
            # SSS está en el símbolo 2 del SSB (0-3)
            # En el grid_ssb extraído (símbolos 2-5 del original), SSS está en posición 0
            # Estructura de grid_ssb: [PBCH+DMRS(sym2), PSS(sym3), PBCH+DMRS(sym4), SSS(sym5)]
            # Pero según 3GPP, la estructura SSB es:
            # Símbolo 0: PBCH+DMRS, Símbolo 1: PSS, Símbolo 2: PBCH+DMRS, Símbolo 3: SSS
            # Si rxGrid(:, 2:5) en MATLAB extrae símbolos 2,3,4,5 (1-indexed)
            # Entonces en 0-indexed son símbolos 1,2,3,4
            # El SSS estaría en símbolo 3 del grid SSB extraído
            sss_symbol = grid_ssb[:, 3]  # Último símbolo del grid SSB
            
            for nid1_test in range(336):
                ncellid_test = 3 * nid1_test + nid2
                sss_ref = self._generate_sss(ncellid_test)
                
                # Extraer región central (127 subportadoras centrales)
                center = len(sss_symbol) // 2
                half_len = len(sss_ref) // 2
                if center + half_len <= len(sss_symbol):
                    sss_rx = sss_symbol[center-half_len:center+half_len+1][:len(sss_ref)]
                    if len(sss_rx) == len(sss_ref):
                        # Correlación
                        corr = np.abs(np.sum(sss_rx * np.conj(sss_ref)))**2
                        
                        if corr > max_corr:
                            max_corr = corr
                            best_nid1 = nid1_test
        
        ncellid = 3 * best_nid1 + nid2
        
        # Determinar SSB más fuerte (simplificado)
        strongest_ssb_idx = 0
        detected = max_corr > 1e-3
        
        if verbose:
            print(f"  NID1: {best_nid1}, Cell ID: {ncellid}, Max corr: {max_corr:.2e}, Detected: {detected}")
        
        # El grid ya está demodulado (grid_full)
        resource_grid = grid_full
        
        # Calcular posición del SSB en el resource grid
        # SSB debe aparecer en símbolos 2-5 (índices 1-4 en 0-indexed)
        nrb_ssb = 20
        demod_rb = 45
        ssb_freq_origin = 12 * (demod_rb - nrb_ssb) // 2 + 1  # Subportadora inicial
        start_symbol = 1  # SSB en símbolos 2-5 -> empieza en índice 1 (símbolo 2)
        num_symbols_ssb = 4
        
        ssb_info = {
            'rect': [start_symbol + 0.5, ssb_freq_origin - 0.5, num_symbols_ssb, 12*nrb_ssb],
            'text': f'SSB idx:{strongest_ssb_idx} | Cell ID:{ncellid}',
            'ncellid': ncellid,
            'nid2': nid2,
            'nid1': best_nid1,
            'freq_offset': freq_offset,
            'timing_offset': timing_offset,
            'correlation': max_corr
        }
        
        return detected, resource_grid, ncellid, strongest_ssb_idx, ssb_info
    
    @staticmethod
    def estimate_snr(signal_data: np.ndarray) -> float:
        """
        Estima SNR de la señal
        
        Args:
            signal_data: Señal compleja
            
        Returns:
            SNR en dB
        """
        noise_est = np.median(np.abs(signal_data))**2
        signal_est = np.mean(np.abs(signal_data)**2)
        snr_db = 10 * np.log10(signal_est / (noise_est + 1e-10))
        return snr_db
    
    @staticmethod
    def calculate_power(signal_data: np.ndarray) -> float:
        """
        Calcula potencia de la señal en dB
        
        Args:
            signal_data: Señal compleja
            
        Returns:
            Potencia en dB
        """
        power_db = 10 * np.log10(np.mean(np.abs(signal_data)**2) + 1e-10)
        return power_db


def capture_waveforms(rx: USRPB210Receiver, monitor_time: float, 
                     interval: float, frames_per_capture: int) -> Tuple[List[np.ndarray], List[float]]:
    """
    Captura múltiples waveforms a intervalos regulares
    
    Args:
        rx: Receptor USRP
        monitor_time: Tiempo total de monitoreo en segundos
        interval: Intervalo entre capturas en segundos
        frames_per_capture: Número de frames por captura
        
    Returns:
        waveforms_all: Lista de waveforms capturados
        capture_times: Tiempos de captura de cada waveform
    """
    capture_duration = (frames_per_capture + 1) * 10e-3  # 10ms por frame
    num_captures = int(monitor_time / interval)
    waveforms_all = []
    capture_times = []
    
    print(f'\nCapturando {num_captures} waveforms...')
    
    for k in range(num_captures):
        t_start = time.time()
        waveform = rx.capture(capture_duration)
        t_elapsed = time.time() - t_start
        
        waveforms_all.append(waveform)
        capture_times.append(t_elapsed)
        
        print(f'[Captura {k+1}/{num_captures}] Tiempo: {t_elapsed:.3f}s, Muestras: {len(waveform)}')
        
        if k < num_captures - 1:
            time.sleep(interval)
    
    return waveforms_all, capture_times


def demodulate_all(waveforms_all: List[np.ndarray], processor: NR5GProcessor, 
                   base_time: float = 0.0, n_symbols_display: int = 14) -> dict:
    """
    Demodula todos los waveforms capturados
    
    Args:
        waveforms_all: Lista de waveforms
        processor: Procesador 5G NR
        base_time: Tiempo base para timestamps
        n_symbols_display: Número de símbolos OFDM a mostrar
        
    Returns:
        Diccionario con todos los resultados de demodulación
    """
    num_captures = len(waveforms_all)
    
    resource_grids = []
    demod_times = []
    ssb_times = []
    power_vec = []
    snr_vec = []
    cellid_vec = []
    ssb_infos = []
    
    print(f'\nDemodulando {num_captures} waveforms...')
    
    for k, waveform in enumerate(waveforms_all):
        t_start = time.time()
        
        # Detectar SSB y demodular (verbose en primera captura)
        verbose = (k == 0)
        detected, grid, ncellid, ssb_idx, ssb_info = processor.find_ssb(waveform, verbose=verbose, n_symbols_display=n_symbols_display)
        
        # Calcular métricas
        power = processor.calculate_power(waveform)
        snr = processor.estimate_snr(waveform)
        
        t_elapsed = time.time() - t_start
        
        resource_grids.append(grid)
        demod_times.append(t_elapsed)
        ssb_times.append(base_time + k * 0.16)
        power_vec.append(power)
        snr_vec.append(snr)
        cellid_vec.append(ncellid)
        ssb_infos.append(ssb_info)
        
        status = '✓' if detected else '✗'
        print(f'[Demod {k+1}/{num_captures}] {status} Tiempo: {t_elapsed:.3f}s | '
              f'Pot={power:.1f}dB | SNR={snr:.1f}dB | cellID={ncellid} | '
              f'Corr={ssb_info.get("correlation", 0):.2e}')
    
    return {
        'resource_grids': resource_grids,
        'demod_times': demod_times,
        'ssb_times': ssb_times,
        'power_vec': power_vec,
        'snr_vec': snr_vec,
        'cellid_vec': cellid_vec,
        'ssb_infos': ssb_infos
    }


def visualize_resource_grids(resource_grids: List[np.ndarray], ssb_times: List[float],
                            center_freq: float, ssb_infos: List[dict]):
    """
    Visualiza la evolución de los resource grids con control interactivo
    
    Args:
        resource_grids: Lista de resource grids
        ssb_times: Tiempos de cada captura
        center_freq: Frecuencia central en Hz
        ssb_infos: Información de SSB para cada captura
    """
    num_captures = len(resource_grids)
    
    # Filtrar grids válidos
    valid_grids = [g for g in resource_grids if g is not None]
    if not valid_grids:
        print("No hay resource grids válidos para visualizar")
        return
    
    # Encontrar dimensiones máximas
    max_rows = max(g.shape[0] for g in valid_grids)
    max_cols = max(g.shape[1] for g in valid_grids)
    
    # Normalizar todos los grids al mismo tamaño
    all_grids = np.zeros((max_rows, max_cols, num_captures))
    for k, grid in enumerate(resource_grids):
        if grid is not None:
            rows, cols = grid.shape
            all_grids[:rows, :cols, k] = np.abs(grid)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.15)
    
    # Mostrar primer grid
    im = ax.imshow(all_grids[:, :, 0], aspect='auto', cmap='jet', 
                   origin='lower', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Magnitude')
    
    ax.set_xlabel('OFDM Symbol')
    ax.set_ylabel('Subcarrier')
    ax.set_title(f'Resource Grid at t = {ssb_times[0]:.2f}s ({center_freq/1e6:.2f} MHz) - [1/{num_captures}]')
    
    # Dibujar rectángulo SSB
    rect = plt.Rectangle((ssb_infos[0]['rect'][0], ssb_infos[0]['rect'][1]),
                         ssb_infos[0]['rect'][2], ssb_infos[0]['rect'][3],
                         linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    # Texto SSB
    text = ax.text(ssb_infos[0]['rect'][0], ssb_infos[0]['rect'][1] + 20,
                   ssb_infos[0]['text'], color='white', fontsize=12)
    
    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_captures-1, 
                   valinit=0, valstep=1)
    
    def update(val):
        idx = int(slider.val)
        im.set_data(all_grids[:, :, idx])
        
        # Actualizar rectángulo y texto
        rect.set_xy((ssb_infos[idx]['rect'][0], ssb_infos[idx]['rect'][1]))
        rect.set_width(ssb_infos[idx]['rect'][2])
        rect.set_height(ssb_infos[idx]['rect'][3])
        
        text.set_position((ssb_infos[idx]['rect'][0], ssb_infos[idx]['rect'][1] + 20))
        text.set_text(ssb_infos[idx]['text'])
        
        ax.set_title(f'Resource Grid at t = {ssb_times[idx]:.2f}s ({center_freq/1e6:.2f} MHz) - [{idx+1}/{num_captures}]')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Navegación con teclado
    def on_key(event):
        idx = int(slider.val)
        if event.key == 'right':
            idx = min(idx + 1, num_captures - 1)
        elif event.key == 'left':
            idx = max(idx - 1, 0)
        else:
            return
        slider.set_val(idx)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()


def parse_arguments():
    """
    Procesa argumentos de línea de comandos
    
    Returns:
        Namespace con los argumentos procesados
    """
    parser = argparse.ArgumentParser(
        description='Monitoreo Continuo de Señales 5G NR usando USRP B210',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos de uso:
  # Usar archivo de configuración
  %(prog)s --config config.yaml
  
  # Listar dispositivos disponibles
  %(prog)s --list-devices
  
  # Usar dispositivo por índice
  %(prog)s --device-index 0
  
  # Usar dispositivo por número de serie
  %(prog)s --device-serial 12345678
  
  # Cambiar frecuencia (GSCN) sobrescribiendo config
  %(prog)s --config config.yaml --gscn 7880
  
  # Ajustar ganancia
  %(prog)s --gain 40
  
  # Modo simulación (sin hardware)
  %(prog)s --simulate
  
NOTA: Los argumentos de línea de comandos tienen prioridad sobre el archivo de configuración
        '''
    )
    
    # Configuración
    parser.add_argument('--config', type=str, metavar='FILE',
                       help='Archivo de configuración YAML (default: config.yaml)')
    
    # Dispositivo
    device_group = parser.add_argument_group('Selección de dispositivo')
    device_group.add_argument('--list-devices', action='store_true',
                              help='Listar dispositivos USRP disponibles y salir')
    device_group.add_argument('--device-index', type=int, metavar='N',
                              help='Índice del dispositivo a usar (0, 1, 2, ...)')
    device_group.add_argument('--device-serial', type=str, metavar='SERIAL',
                              help='Número de serie del dispositivo a usar')
    device_group.add_argument('--device-args', type=str, metavar='ARGS',
                              help='Argumentos adicionales del dispositivo')
    
    # Configuración RF
    rf_group = parser.add_argument_group('Configuración RF')
    rf_group.add_argument('--gscn', type=int, metavar='N',
                         help='Global Synchronization Channel Number')
    rf_group.add_argument('--sample-rate', type=float, metavar='Hz',
                         help='Tasa de muestreo en Hz')
    rf_group.add_argument('--gain', type=float, metavar='dB',
                         help='Ganancia del receptor en dB')
    rf_group.add_argument('--scs', type=int, choices=[15, 30, 60, 120, 240],
                         help='Subcarrier spacing en kHz')
    rf_group.add_argument('--n-symbols-display', type=int, metavar='N',
                         help='Número de símbolos OFDM a mostrar en gráfica (6-14)')
    
    # Parámetros de monitoreo
    monitor_group = parser.add_argument_group('Parámetros de monitoreo')
    monitor_group.add_argument('--monitor-time', type=float, metavar='s',
                              help='Tiempo total de monitoreo en segundos')
    monitor_group.add_argument('--interval', type=float, metavar='s',
                              help='Intervalo entre capturas en segundos')
    monitor_group.add_argument('--frames', type=int, metavar='N',
                              help='Número de frames por captura')
    
    # Modo simulación
    parser.add_argument('--simulate', action='store_true',
                       help='Modo simulación (genera datos sintéticos sin hardware)')
    
    # Visualización
    parser.add_argument('--no-gui', action='store_true',
                       help='Desactivar visualización gráfica')
    parser.add_argument('--verbose', action='store_true',
                       help='Mostrar información detallada de procesado')
    
    return parser.parse_args()


def main():
    """Función principal"""
    args = parse_arguments()
    
    # Si solo se pide listar dispositivos
    if args.list_devices:
        list_usrp_devices()
        return
    
    # Cargar configuración desde archivo
    config_file = args.config if args.config else 'config.yaml'
    config = load_config(config_file)
    
    # Fusionar con argumentos de línea de comandos
    config = merge_config(config, args)
    
    print('=== INICIANDO MONITOREO CONTINUO 5G NR ===\n')
    
    # Extraer parámetros de configuración
    gscn = get_config_value(config, 'rf.gscn', 7929)
    sample_rate = get_config_value(config, 'rf.sample_rate', 19.5e6)
    gain = get_config_value(config, 'rf.gain', 50)
    scs = get_config_value(config, 'rf.scs', 30)
    
    n_symbols_display = get_config_value(config, 'processing.n_symbols_display', 14)
    
    monitor_time = get_config_value(config, 'monitoring.monitor_time', 0.57)
    interval = get_config_value(config, 'monitoring.interval', 0.057)
    frames_per_capture = get_config_value(config, 'monitoring.frames_per_capture', 1)
    
    enable_gui = get_config_value(config, 'visualization.enable_gui', True)
    verbose = get_config_value(config, 'visualization.verbose', False)
    
    simulate = get_config_value(config, 'simulation.enabled', False)
    
    device_index = get_config_value(config, 'device.index', None)
    device_serial = get_config_value(config, 'device.serial', None)
    device_args = get_config_value(config, 'device.args', '')
    
    # Configuración
    center_freq = NR5GProcessor.gscn_to_frequency(gscn)
    
    print(f'Configuración:')
    if args.config:
        print(f'  Archivo config: {config_file}')
    print(f'  GSCN: {gscn}')
    print(f'  Frecuencia central: {center_freq/1e6:.2f} MHz')
    print(f'  Tasa de muestreo: {sample_rate/1e6:.2f} MHz')
    print(f'  Ganancia: {gain} dB')
    print(f'  SCS: {scs} kHz')
    print(f'  Símbolos OFDM a mostrar: {n_symbols_display}')
    print(f'  Tiempo de monitoreo: {monitor_time}s')
    print(f'  Intervalo entre capturas: {interval}s')
    print(f'  Frames por captura: {frames_per_capture}')
    print(f'  Modo: {"SIMULACIÓN" if simulate else "HARDWARE"}')
    print(f'  GUI: {"Activada" if enable_gui else "Desactivada"}')
    if verbose:
        print(f'  Verbose: Activado')
    
    # Inicializar procesador
    processor = NR5GProcessor(scs, sample_rate)
    
    # Modo simulación
    if simulate:
        print('\n⚠ MODO SIMULACIÓN: Generando datos sintéticos...')
        num_captures = int(monitor_time / interval)
        capture_duration = (frames_per_capture + 1) * 10e-3
        num_samples = int(capture_duration * sample_rate)
        
        waveforms_all = []
        for _ in range(num_captures):
            # Generar ruido complejo
            waveform = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
            waveforms_all.append(waveform)
        
        results = demodulate_all(waveforms_all, processor, n_symbols_display=n_symbols_display)
        
        if enable_gui:
            visualize_resource_grids(
                results['resource_grids'],
                results['ssb_times'],
                center_freq,
                results['ssb_infos']
            )
        
        print('\n=== SIMULACIÓN COMPLETADA ===')
        return
    
    # Modo hardware
    try:
        # Seleccionar dispositivo
        selected_device_args = select_usrp_device(
            device_index=device_index,
            device_serial=device_serial,
            device_args=device_args
        )
        
        # Inicializar receptor USRP
        print('\n--- Inicializando USRP B210 ---')
        rx = USRPB210Receiver(center_freq, sample_rate, gain, device_args=selected_device_args)
        
    except Exception as e:
        print(f'\n❌ Error al inicializar USRP: {e}')
        print('\nPuedes usar --list-devices para ver dispositivos disponibles')
        print('o --simulate para ejecutar sin hardware.')
        return
    
    try:
        # PASO 1: Captura
        print('\n--- PASO 1: CAPTURA ---')
        t_capture_start = time.time()
        waveforms_all, capture_times = capture_waveforms(
            rx, monitor_time, interval, frames_per_capture
        )
        t_capture_total = time.time() - t_capture_start
        
        # PASO 2: Demodulación
        print('\n--- PASO 2: DEMODULACIÓN ---')
        t_demod_start = time.time()
        results = demodulate_all(waveforms_all, processor, n_symbols_display=n_symbols_display)
        t_demod_total = time.time() - t_demod_start
        
        # PASO 3: Visualización
        if enable_gui:
            print('\n--- PASO 3: VISUALIZACIÓN ---')
            t_vis_start = time.time()
            visualize_resource_grids(
                results['resource_grids'],
                results['ssb_times'],
                center_freq,
                results['ssb_infos']
            )
            t_vis_total = time.time() - t_vis_start
        else:
            t_vis_total = 0
        
        # Reporte final
        print('\n=== RESUMEN DE TIEMPOS ===')
        print(f'Captura promedio: {np.mean(capture_times):.3f}s (total: {t_capture_total:.3f}s)')
        print(f'Demodulación promedio: {np.mean(results["demod_times"]):.3f}s (total: {t_demod_total:.3f}s)')
        if enable_gui:
            print(f'Visualización: {t_vis_total:.3f}s')
        
        print('\n=== MÉTRICAS DE SEÑAL ===')
        print(f'Potencia promedio: {np.mean(results["power_vec"]):.1f} dB')
        print(f'SNR promedio: {np.mean(results["snr_vec"]):.1f} dB')
        print(f'Cell IDs detectados: {set(results["cellid_vec"])}')
        
    finally:
        # Liberar recursos
        print('\n--- Liberando recursos USRP ---')
        rx.release()
        print('\n=== PROCESO COMPLETADO ===')


if __name__ == '__main__':
    main()
