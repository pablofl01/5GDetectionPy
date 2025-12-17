#!/usr/bin/env python3
"""
Frequency correction module for 5G NR signals.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple

from py3gpp.nrPSS import nrPSS
from py3gpp.nrPSSIndices import nrPSSIndices
from py3gpp.nrOFDMModulate import nrOFDMModulate


def frequency_correction_ofdm(waveform: np.ndarray, scs: int, sample_rate: float, 
                               search_bw: float, verbose: bool = False) -> Tuple[np.ndarray, float, int]:
    """
    Frequency correction and PSS detection using OFDM modulation.
    
    Args:
        waveform: Captured IQ signal
        scs: Subcarrier spacing in kHz (typically 30)
        sample_rate: Sample rate in Hz (typically 19.5e6)
        search_bw: Search bandwidth in kHz (typically 3*scs = 90)
        verbose: Display detailed processing information
    
    Returns:
        waveform_corrected: Waveform with frequency correction
        freq_offset: Detected frequency offset in Hz
        nid2: Detected PSS ID (0, 1 or 2)
    """
    if verbose:
        print("Frequency correction and PSS detection (OFDM method)...")
    
    # Synchronization parameters
    sync_nfft = 256
    sync_sr = sync_nfft * scs * 1000  # 256 * 30 * 1000 = 7.68 MHz
    nrb_ssb = 20  # SSB is 20 RBs = 240 subcarriers
    
    # PSS indices
    pss_indices = nrPSSIndices()
    
    # Create reference grids for the 3 NID2
    ref_grids = np.zeros((nrb_ssb * 12, 4, 3), dtype=complex)
    for nid2 in range(3):
        ref_grids[pss_indices, 0, nid2] = nrPSS(nid2)
    
    # Coarse and fine search
    coarse_fshifts = np.arange(-search_bw, search_bw + scs, scs) * 1e3 / 2
    fine_fshifts = np.arange(-scs, scs + 1, 1) * 1e3 / 2
    fshifts = np.unique(np.concatenate([coarse_fshifts, fine_fshifts]))
    fshifts = np.sort(fshifts)
    
    peak_values = np.zeros((len(fshifts), 3))
    t = np.arange(len(waveform)) / sample_rate
    
    if verbose:
        print(f"  Testing {len(fshifts)} frequency offsets × 3 NID2...")
    
    for f_idx, f_shift in enumerate(fshifts):
        waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * f_shift * t)
        num_samples_ds = int(len(waveform_corrected) * sync_sr / sample_rate)
        waveform_ds = scipy_signal.resample(waveform_corrected, num_samples_ds)
        
        for nid2 in range(3):
            try:
                ref_grid_nid2 = ref_grids[:, :, nid2]
                ref_waveform, _ = nrOFDMModulate(
                    grid=ref_grid_nid2,
                    scs=scs,
                    initialNSlot=0,
                    SampleRate=sync_sr,
                    Nfft=sync_nfft
                )
                
                max_samples = min(len(waveform_ds), 300000)
                corr = scipy_signal.correlate(waveform_ds[:max_samples], 
                                            ref_waveform, mode='valid')
                peak_values[f_idx, nid2] = np.max(np.abs(corr))
            except Exception as e:
                peak_values[f_idx, nid2] = 0
    
    # Normalize and display results
    peak_values_norm = peak_values / np.max(peak_values) if np.max(peak_values) > 0 else peak_values
    
    if verbose:
        print(f"\n  PSS correlation matrix (normalized):")
        print(f"  {'Freq (kHz)':>12} {'NID2=0':>12} {'NID2=1':>12} {'NID2=2':>12}")
        for i, f in enumerate(fshifts):
            print(f"  {f/1e3:>12.2f} {peak_values_norm[i, 0]:>12.3f} {peak_values_norm[i, 1]:>12.3f} {peak_values_norm[i, 2]:>12.3f}")
    
    best_f_idx, best_nid2 = np.unravel_index(np.argmax(peak_values), peak_values.shape)
    freq_offset = fshifts[best_f_idx]
    
    if verbose:
        print(f"\n  → Detected NID2: {best_nid2}")
        print(f"  → Frequency offset: {freq_offset/1e3:.3f} kHz")
        print(f"  → Maximum peak: {peak_values[best_f_idx, best_nid2]:.2f}")
    
    waveform_corrected = waveform * np.exp(-1j * 2 * np.pi * freq_offset * t)
    
    return waveform_corrected, freq_offset, best_nid2
