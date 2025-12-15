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
    merged = {
        'device': config.get('device', {}),
        'rf': config.get('rf', {}),
        'processing': config.get('processing', {}),
        'monitoring': config.get('monitoring', {}),
        'visualization': config.get('visualization', {}),
        'simulation': config.get('simulation', {}),
        'export': config.get('export', {})
    }

    if hasattr(args, 'device_index') and args.device_index is not None:
        merged['device']['index'] = args.device_index
    if hasattr(args, 'device_serial') and args.device_serial is not None:
        merged['device']['serial'] = args.device_serial
    if hasattr(args, 'device_args') and args.device_args:
        merged['device']['args'] = args.device_args

    if hasattr(args, 'gscn') and args.gscn is not None:
        merged['rf']['gscn'] = args.gscn
    if hasattr(args, 'sample_rate') and args.sample_rate is not None:
        merged['rf']['sample_rate'] = args.sample_rate
    if hasattr(args, 'gain') and args.gain is not None:
        merged['rf']['gain'] = args.gain
    if hasattr(args, 'scs') and args.scs is not None:
        merged['rf']['scs'] = args.scs

    if hasattr(args, 'n_symbols_display') and args.n_symbols_display is not None:
        merged['processing']['n_symbols_display'] = args.n_symbols_display

    if hasattr(args, 'monitor_time') and args.monitor_time is not None:
        merged['monitoring']['monitor_time'] = args.monitor_time
    if hasattr(args, 'interval') and args.interval is not None:
        merged['monitoring']['interval'] = args.interval
    if hasattr(args, 'frames') and args.frames is not None:
        merged['monitoring']['frames_per_capture'] = args.frames

    if hasattr(args, 'no_gui') and args.no_gui:
        merged['visualization']['enable_gui'] = False
    if hasattr(args, 'verbose') and args.verbose:
        merged['visualization']['verbose'] = True

    if hasattr(args, 'simulate') and args.simulate:
        merged['simulation']['enabled'] = True

    return merged


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    keys = path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value if value is not None else default


def list_usrp_devices():
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

    print('\n' + '=' * 40)
    return devices


def select_usrp_device(device_index: Optional[int] = None,
                       device_serial: Optional[str] = None,
                       device_args: str = "") -> str:
    devices = list_usrp_devices()

    if not devices:
        raise RuntimeError("No hay dispositivos USRP disponibles")

    if device_index is not None:
        if 0 <= device_index < len(devices):
            selected = devices[device_index]
            print(f'\n✓ Seleccionado dispositivo [{device_index}]: {selected.get("serial", "N/A")}')
            if 'serial' in selected:
                return f"serial={selected['serial']},{device_args}" if device_args else f"serial={selected['serial']}"
            return device_args
        else:
            raise ValueError(f"Índice {device_index} fuera de rango. Hay {len(devices)} dispositivos.")

    if device_serial is not None:
        for dev in devices:
            if dev.get('serial') == device_serial:
                print(f'\n✓ Seleccionado dispositivo con serial: {device_serial}')
                return f"serial={device_serial},{device_args}" if device_args else f"serial={device_serial}"
        raise ValueError(f"No se encontró dispositivo con serial: {device_serial}")

    if len(devices) == 1:
        selected = devices[0]
        print(f'\n✓ Usando único dispositivo disponible: {selected.get("serial", "N/A")}')
        if 'serial' in selected:
            return f"serial={selected['serial']},{device_args}" if device_args else f"serial={selected['serial']}"
        return device_args

    print(f'\n⚠ Hay {len(devices)} dispositivos. Especifica --device-index o --device-serial')
    print('   o usa --list-devices para ver la lista completa.')
    raise RuntimeError("Múltiples dispositivos encontrados. Especifica cuál usar.")


class USRPB210Receiver:
    """Clase para manejar el USRP B210"""

    def __init__(self, center_freq: float, sample_rate: float, gain: float,
                 channels=[0], device_args=""):
        self.usrp = uhd.usrp.MultiUSRP(device_args)
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.channels = channels
        self._configure()

    def _configure(self):
        self.usrp.set_rx_rate(self.sample_rate, 0)
        actual_rate = self.usrp.get_rx_rate(0)
        print(f"Tasa de muestreo solicitada: {self.sample_rate/1e6:.2f} MHz")
        print(f"Tasa de muestreo obtenida: {actual_rate/1e6:.2f} MHz")

        tune_request = uhd.types.TuneRequest(self.center_freq)
        self.usrp.set_rx_freq(tune_request, 0)
        actual_freq = self.usrp.get_rx_freq(0)
        print(f"Frecuencia central: {actual_freq/1e6:.2f} MHz")

        self.usrp.set_rx_gain(self.gain, 0)
        actual_gain = self.usrp.get_rx_gain(0)
        print(f"Ganancia: {actual_gain:.1f} dB")

        self.usrp.set_rx_antenna("RX2", 0)
        print(f"Antena: {self.usrp.get_rx_antenna(0)}")

    def capture(self, duration: float) -> np.ndarray:
        num_samples = int(duration * self.sample_rate)
        samples = np.zeros(num_samples, dtype=np.complex64)

        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        stream_args.channels = self.channels
        rx_streamer = self.usrp.get_rx_stream(stream_args)

        recv_buffer = np.zeros((1, 10000), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = num_samples
        stream_cmd.stream_now = True
        rx_streamer.issue_stream_cmd(stream_cmd)

        samples_received = 0
        while samples_received < num_samples:
            num_rx_samps = rx_streamer.recv(recv_buffer, metadata, 1.0)
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                print(f"Error en recepción: {metadata.strerror()}")
                break
            end_idx = min(samples_received + num_rx_samps, num_samples)
            samples[samples_received:end_idx] = recv_buffer[0, :end_idx - samples_received]
            samples_received = end_idx

        return samples

    def release(self):
        self.usrp = None


class NR5GProcessor:
    """Clase para procesar señales 5G NR con timing estilo nrTimingEstimate"""

    def __init__(self, scs_khz: int = 30, sample_rate: float = 19.5e6):
        self.scs_khz = scs_khz
        self.sample_rate = sample_rate

        self.nrb_ssb = 20
        self.n_subcarriers_ssb = self.nrb_ssb * 12  # 240

        # Nfft coherente con sample_rate ≈ Nfft*SCS (30 kHz): 19.5e6/30e3 ≈ 650 → 1024
        self.fft_size_ref = 1024
        # CP normal aproximado: Nfft/14 ≈ 73
        self.cp_samples_ref = 72

    @staticmethod
    def gscn_to_frequency(gscn: int) -> float:
        if 7499 <= gscn <= 22255:
            N = gscn - 7499
            freq_hz = 3000e6 + N * 1.44e6
            return freq_hz
        else:
            raise ValueError(f"GSCN {gscn} fuera de rango FR1")

    def _generate_pss(self, nid2: int) -> np.ndarray:
        x = np.zeros(127, dtype=np.complex128)
        for n in range(127):
            m = (n + 43 * nid2) % 127
            x[n] = np.exp(-1j * np.pi * m * (m + 1) / 63)
        return x

    def _generate_sss(self, ncellid: int) -> np.ndarray:
        nid1 = ncellid // 3
        nid2 = ncellid % 3
        x0 = np.zeros(127, dtype=np.complex128)
        x1 = np.zeros(127, dtype=np.complex128)
        for n in range(127):
            m0 = (n + nid1) % 127
            m1 = (n + nid2) % 127
            x0[n] = np.exp(-1j * np.pi * m0 * (m0 + 1) / 63)
            x1[n] = np.exp(-1j * np.pi * m1 * (m1 + 1) / 63)
        return x0 * x1

    def _fine_frequency_offset(self, waveform: np.ndarray) -> float:
        fft_size = 256
        cp_samples = 18 if self.scs_khz == 30 else 20
        symbol_length = fft_size + cp_samples
        if len(waveform) < 4 * symbol_length:
            return 0.0

        delayed = np.concatenate([np.zeros(fft_size), waveform[:-fft_size]])
        cp_product = waveform * np.conj(delayed)
        cp_xcorr = np.convolve(cp_product, np.ones(cp_samples), mode='same')

        y = cp_xcorr.copy()
        for k in range(1, 4):
            delayed_xcorr = np.concatenate([np.zeros(k * symbol_length),
                                            cp_xcorr[:-(k * symbol_length)]])
            y += delayed_xcorr

        cp_corr_index = fft_size + cp_samples + 3 * symbol_length
        if cp_corr_index < len(y):
            scs_hz = self.scs_khz * 1000
            freq_offset = scs_hz * np.angle(y[cp_corr_index]) / (2 * np.pi)
            return freq_offset
        return 0.0

    def frequency_correction(self, waveform: np.ndarray,
                             search_bw_khz: float = 90) -> Tuple[np.ndarray, float, int]:
        pss_sequences = [self._generate_pss(nid2) for nid2 in range(3)]
        scs_hz = self.scs_khz * 1000
        search_range = search_bw_khz * 1000
        freq_offsets = np.arange(-search_range, search_range + scs_hz / 2, scs_hz / 2)

        max_corr = 0
        best_offset = 0
        best_nid2 = 0

        fft_size = self.fft_size_ref
        cp_samples = self.cp_samples_ref
        symbol_length = fft_size + cp_samples

        for freq_off in freq_offsets:
            t = np.arange(len(waveform)) / self.sample_rate
            corrected = waveform * np.exp(-1j * 2 * np.pi * freq_off * t)

            if len(corrected) < 2 * symbol_length:
                continue

            grid_search = np.zeros((fft_size, 2), dtype=np.complex128)
            for sym_idx in range(2):
                start = sym_idx * symbol_length + cp_samples
                end = start + fft_size
                if end <= len(corrected):
                    symbol = corrected[start:end]
                    freq_domain = fft(symbol)
                    grid_search[:, sym_idx] = fftshift(freq_domain)

            pss_symbol = grid_search[:, 1]

            for nid2, pss_ref in enumerate(pss_sequences):
                center = len(pss_symbol) // 2
                half_len = len(pss_ref) // 2
                start = center - half_len
                end = start + len(pss_ref)
                if start < 0 or end > len(pss_symbol):
                    continue
                pss_rx = pss_symbol[start:end]
                corr = np.abs(np.sum(pss_rx * np.conj(pss_ref))) ** 2
                if corr > max_corr:
                    max_corr = corr
                    best_offset = freq_off
                    best_nid2 = nid2

        t = np.arange(len(waveform)) / self.sample_rate
        waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * best_offset * t)
        fine_offset = self._fine_frequency_offset(waveform_corrected)
        total_offset = best_offset + fine_offset
        waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * total_offset * t)

        return waveform_corrected, total_offset, best_nid2

    def _build_ref_grid_pss(self, nrb: int, nid2: int) -> np.ndarray:
        K = nrb * 12
        L = 14
        P = 1
        ref_grid = np.zeros((K, L, P), dtype=np.complex128)

        pss_seq = self._generate_pss(nid2)
        center = K // 2
        start = center - len(pss_seq) // 2
        end = start + len(pss_seq)
        ref_grid[start:end, 1, 0] = pss_seq  # símbolo 1 (0‑based)

        return ref_grid

    def _ofdm_modulate_ref(self, ref_grid: np.ndarray) -> np.ndarray:
        K, L, P = ref_grid.shape
        fft_size = self.fft_size_ref
        cp_samples = self.cp_samples_ref
        symbol_length = fft_size + cp_samples

        ref_waveform = np.zeros((L * symbol_length, P), dtype=np.complex128)

        for p in range(P):
            for l in range(L):
                freq_sym = np.zeros(fft_size, dtype=np.complex128)
                start_fft = (fft_size - K) // 2
                freq_sym[start_fft:start_fft + K] = ref_grid[:, l, p]
                time_sym = ifft(fftshift(freq_sym)) * fft_size
                cp = time_sym[-cp_samples:]
                ofdm_sym = np.concatenate([cp, time_sym])
                ref_waveform[l * symbol_length:(l + 1) * symbol_length, p] = ofdm_sym

        return ref_waveform

    def _timing_estimate(self, waveform: np.ndarray, nrb: int, nid2: int) -> int:
        ref_grid = self._build_ref_grid_pss(nrb, nid2)
        ref = self._ofdm_modulate_ref(ref_grid)  # T_ref x P

        if waveform.ndim == 1:
            waveform = waveform.reshape(-1, 1)

        T, R = waveform.shape
        minlength = ref.shape[0]

        if T < minlength:
            waveform_pad = np.vstack([waveform,
                                      np.zeros((minlength - T, R), dtype=waveform.dtype)])
            T_pad = minlength
        else:
            waveform_pad = waveform
            T_pad = T

        P = ref.shape[1]
        mag = np.zeros((T_pad, R, P), dtype=np.float64)

        for r in range(R):
            for p in range(P):
                refcorr = signal.correlate(waveform_pad[:, r], ref[:, p], mode='full')
                tail = refcorr[T_pad - 1:]
                mag[:, r, p] = np.abs(tail)

        mag_sum_ports = np.sum(mag, axis=2)
        mag_sum_rx = np.sum(mag_sum_ports, axis=1)

        peakindex = int(np.argmax(mag_sum_rx))
        offset = peakindex  # 0‑based

        return offset

    def ofdm_demodulate(self, waveform: np.ndarray,
                        nrb: int = 45, n_symbols: int = 14) -> Optional[np.ndarray]:
        K = nrb * 12
        fft_size = self.fft_size_ref
        cp_samples = self.cp_samples_ref
        symbol_length = fft_size + cp_samples

        n_samples_needed = n_symbols * symbol_length
        if len(waveform) < n_samples_needed:
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

        center = fft_size // 2
        half = K // 2
        grid_extracted = grid[center - half:center + half, :]

        return grid_extracted

    def find_ssb(self, waveform: np.ndarray, verbose: bool = False,
                 n_symbols_display: int = 14) -> Tuple[bool, Optional[np.ndarray], int, int, dict]:
        if verbose:
            print(f"  Muestras recibidas: {len(waveform)}, "
                  f"Potencia: {10*np.log10(np.mean(np.abs(waveform)**2)+1e-12):.1f} dB")

        corrected_waveform, freq_offset, nid2 = self.frequency_correction(waveform)
        if verbose:
            print(f"  Freq offset: {freq_offset:.1f} Hz, NID2: {nid2}")

        timing_offset = self._timing_estimate(corrected_waveform, self.nrb_ssb, nid2)
        if verbose:
            print(f"  Timing offset: {timing_offset} muestras (inicio del SSB aprox.)")

        if timing_offset > 0 and timing_offset < len(corrected_waveform):
            corrected_waveform = corrected_waveform[timing_offset:]

        if verbose:
            print(f"  Señal recortada: {len(corrected_waveform)} muestras")
            print("  SSB ahora empieza cerca de la muestra 0")

        grid_ssb_20rb = self.ofdm_demodulate(corrected_waveform,
                                             nrb=self.nrb_ssb,
                                             n_symbols=14)

        if grid_ssb_20rb is None or grid_ssb_20rb.shape[1] < 4:
            if verbose:
                print("  ❌ No se pudo demodular suficientes símbolos OFDM")
            return False, None, 0, 0, {}

        grid_ssb = grid_ssb_20rb[:, 0:4]

        max_corr = 0
        best_nid1 = 0

        if grid_ssb.shape[1] >= 4:
            sss_symbol = grid_ssb[:, 3]
            for nid1_test in range(336):
                ncellid_test = 3 * nid1_test + nid2
                sss_ref = self._generate_sss(ncellid_test)
                center = len(sss_symbol) // 2
                half_len = len(sss_ref) // 2
                start = center - half_len
                end = start + len(sss_ref)
                if start < 0 or end > len(sss_symbol):
                    continue
                sss_rx = sss_symbol[start:end]
                corr = np.abs(np.mean(sss_rx * np.conj(sss_ref))) ** 2
                if corr > max_corr:
                    max_corr = corr
                    best_nid1 = nid1_test

        ncellid = 3 * best_nid1 + nid2
        strongest_ssb_idx = 0
        detected = max_corr > 1e-6

        if verbose:
            print(f"  NID1: {best_nid1}, Cell ID: {ncellid}, "
                  f"SSS corr: {max_corr:.2e}, Detected: {detected}")

        demod_rb = 30
        grid_full = self.ofdm_demodulate(corrected_waveform,
                                         nrb=demod_rb,
                                         n_symbols=n_symbols_display)

        if grid_full is None:
            grid_full = grid_ssb_20rb

        resource_grid = grid_full

        ssb_freq_origin = 12 * (demod_rb - self.nrb_ssb) // 2 + 1
        start_symbol = 0
        num_symbols_ssb = 4

        ssb_info = {
            'rect': [start_symbol + 0.5,
                     ssb_freq_origin - 0.5,
                     num_symbols_ssb,
                     12 * self.nrb_ssb],
            'text': f'SSB idx:{strongest_ssb_idx} | Cell ID:{ncellid}',
            'ncellid': ncellid,
            'nid2': nid2,
            'nid1': best_nid1,
            'freq_offset': freq_offset,
            'timing_offset': timing_offset,
            'correlation': max_corr,
            'ssb_grid': grid_ssb
        }

        return detected, resource_grid, ncellid, strongest_ssb_idx, ssb_info

    @staticmethod
    def estimate_snr(signal_data: np.ndarray) -> float:
        noise_est = np.median(np.abs(signal_data)) ** 2
        signal_est = np.mean(np.abs(signal_data) ** 2)
        snr_db = 10 * np.log10(signal_est / (noise_est + 1e-10))
        return snr_db

    @staticmethod
    def calculate_power(signal_data: np.ndarray,
                        reference_grid: Optional[np.ndarray] = None) -> float:
        if reference_grid is not None:
            max_power = np.max(np.abs(reference_grid) ** 2)
            signal_power = np.mean(np.abs(signal_data) ** 2)
            power_db = 10 * np.log10(signal_power / (max_power + 1e-10))
        else:
            power_db = 10 * np.log10(np.mean(np.abs(signal_data) ** 2) + 1e-10)
        return power_db


def capture_waveforms(rx: USRPB210Receiver, monitor_time: float,
                      interval: float, frames_per_capture: int) -> Tuple[List[np.ndarray], List[float]]:
    capture_duration = (frames_per_capture + 1) * 10e-3
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

        verbose = (k == 0)
        detected, grid, ncellid, ssb_idx, ssb_info = processor.find_ssb(
            waveform, verbose=verbose, n_symbols_display=n_symbols_display
        )

        ssb_grid = ssb_info.get('ssb_grid', None) if ssb_info else None
        power = processor.calculate_power(waveform, reference_grid=ssb_grid)
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
        corr_val = ssb_info.get("correlation", 0) if ssb_info else 0
        print(f'[Demod {k+1}/{num_captures}] {status} Tiempo: {t_elapsed:.3f}s | '
              f'Pot={power:.1f}dB | SNR={snr:.1f}dB | cellID={ncellid} | '
              f'Corr={corr_val:.2e}')

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
    num_captures = len(resource_grids)

    valid_grids = [g for g in resource_grids if g is not None]
    if not valid_grids:
        print("No hay resource grids válidos para visualizar")
        return

    max_rows = max(g.shape[0] for g in valid_grids)
    max_cols = max(g.shape[1] for g in valid_grids)

    all_grids = np.zeros((max_rows, max_cols, num_captures))
    for k, grid in enumerate(resource_grids):
        if grid is not None:
            rows, cols = grid.shape
            all_grids[:rows, :cols, k] = np.abs(grid)

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.15)

    im = ax.imshow(all_grids[:, :, 0], aspect='auto', cmap='jet',
                   origin='lower', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Magnitude')

    ax.set_xlabel('OFDM Symbol')
    ax.set_ylabel('Subcarrier')
    ax.set_title(f'Resource Grid at t = {ssb_times[0]:.2f}s ({center_freq/1e6:.2f} MHz) - [1/{num_captures}]')

    rect = plt.Rectangle((ssb_infos[0]['rect'][0], ssb_infos[0]['rect'][1]),
                         ssb_infos[0]['rect'][2], ssb_infos[0]['rect'][3],
                         linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    text = ax.text(ssb_infos[0]['rect'][0], ssb_infos[0]['rect'][1] + 20,
                   ssb_infos[0]['text'], color='white', fontsize=12)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_captures - 1,
                    valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        im.set_data(all_grids[:, :, idx])

        rect.set_xy((ssb_infos[idx]['rect'][0], ssb_infos[idx]['rect'][1]))
        rect.set_width(ssb_infos[idx]['rect'][2])
        rect.set_height(ssb_infos[idx]['rect'][3])

        text.set_position((ssb_infos[idx]['rect'][0], ssb_infos[idx]['rect'][1] + 20))
        text.set_text(ssb_infos[idx]['text'])

        ax.set_title(f'Resource Grid at t = {ssb_times[idx]:.2f}s ({center_freq/1e6:.2f} MHz) - [{idx+1}/{num_captures}]')
        fig.canvas.draw_idle()

    slider.on_changed(update)

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
    parser = argparse.ArgumentParser(
        description='Monitoreo Continuo de Señales 5G NR usando USRP B210',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos de uso:
  %(prog)s --config config.yaml
  %(prog)s --list-devices
  %(prog)s --device-index 0
  %(prog)s --device-serial 12345678
  %(prog)s --config config.yaml --gscn 7880
  %(prog)s --gain 40
  %(prog)s --simulate

NOTA: Los argumentos de línea de comandos tienen prioridad sobre el archivo de configuración
        '''
    )

    parser.add_argument('--config', type=str, metavar='FILE',
                        help='Archivo de configuración YAML (default: config.yaml)')

    device_group = parser.add_argument_group('Selección de dispositivo')
    device_group.add_argument('--list-devices', action='store_true',
                              help='Listar dispositivos USRP disponibles y salir')
    device_group.add_argument('--device-index', type=int, metavar='N',
                              help='Índice del dispositivo a usar (0, 1, 2, ...)')
    device_group.add_argument('--device-serial', type=str, metavar='SERIAL',
                              help='Número de serie del dispositivo a usar')
    device_group.add_argument('--device-args', type=str, metavar='ARGS',
                              help='Argumentos adicionales del dispositivo')

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

    monitor_group = parser.add_argument_group('Parámetros de monitoreo')
    monitor_group.add_argument('--monitor-time', type=float, metavar='s',
                               help='Tiempo total de monitoreo en segundos')
    monitor_group.add_argument('--interval', type=float, metavar='s',
                               help='Intervalo entre capturas en segundos')
    monitor_group.add_argument('--frames', type=int, metavar='N',
                               help='Número de frames por captura')

    parser.add_argument('--simulate', action='store_true',
                        help='Modo simulación (genera datos sintéticos sin hardware)')

    parser.add_argument('--no-gui', action='store_true',
                        help='Desactivar visualización gráfica')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostrar información detallada de procesado')

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.list_devices:
        list_usrp_devices()
        return

    config_file = args.config if args.config else 'config.yaml'
    config = load_config(config_file)
    config = merge_config(config, args)

    print('=== INICIANDO MONITOREO CONTINUO 5G NR ===\n')

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

    center_freq = NR5GProcessor.gscn_to_frequency(gscn)

    print('Configuración:')
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
        print('  Verbose: Activado')

    processor = NR5GProcessor(scs, sample_rate)

    if simulate:
        print('\n⚠ MODO SIMULACIÓN: Generando datos sintéticos...')
        num_captures = int(monitor_time / interval)
        capture_duration = (frames_per_capture + 1) * 10e-3
        num_samples = int(capture_duration * sample_rate)

        waveforms_all = []
        for _ in range(num_captures):
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

    try:
        selected_device_args = select_usrp_device(
            device_index=device_index,
            device_serial=device_serial,
            device_args=device_args
        )

        print('\n--- Inicializando USRP B210 ---')
        rx = USRPB210Receiver(center_freq, sample_rate, gain, device_args=selected_device_args)

    except Exception as e:
        print(f'\n❌ Error al inicializar USRP: {e}')
        print('\nPuedes usar --list-devices para ver dispositivos disponibles')
        print('o --simulate para ejecutar sin hardware.')
        return

    try:
        print('\n--- PASO 1: CAPTURA ---')
        t_capture_start = time.time()
        waveforms_all, capture_times = capture_waveforms(
            rx, monitor_time, interval, frames_per_capture
        )
        t_capture_total = time.time() - t_capture_start

        print('\n--- PASO 2: DEMODULACIÓN ---')
        t_demod_start = time.time()
        results = demodulate_all(waveforms_all, processor, n_symbols_display=n_symbols_display)
        t_demod_total = time.time() - t_demod_start

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
        print('\n--- Liberando recursos USRP ---')
        rx.release()
        print('\n=== PROCESO COMPLETADO ===')


if __name__ == '__main__':
    main()
