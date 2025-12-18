#!/usr/bin/env python3
"""
Timing estimation module for 5G NR signals.
"""

import numpy as np

from py3gpp.nrPSS import nrPSS
from py3gpp.nrPSSIndices import nrPSSIndices
from py3gpp.nrTimingEstimate import nrTimingEstimate


def estimate_timing_offset(waveform: np.ndarray, nid2: int, scs: int, 
                           sample_rate: float, verbose: bool = False) -> int:
    """
    Timing offset estimation using nrTimingEstimate.
    
    Args:
        waveform: IQ signal with frequency correction applied
        nid2: Detected PSS ID (0, 1 or 2)
        scs: Subcarrier spacing in kHz
        sample_rate: Sample rate in Hz
        verbose: If True, displays processing information
    
    Returns:
        timing_offset: Offset in samples from the start of the slot
    """
    if verbose:
        print("Timing offset estimation...")
    
    nrb_ssb = 20
    pss_indices = nrPSSIndices()
    pss_seq = nrPSS(nid2)
    
    # Create refGrid with PSS in symbol 2 (0-indexed: symbol 1)
    ref_grid = np.zeros((nrb_ssb * 12, 2), dtype=complex)
    ref_grid[pss_indices.astype(int), 1] = pss_seq
    
    timing_offset = nrTimingEstimate(
        waveform=waveform,
        nrb=nrb_ssb,
        scs=scs,
        initialNSlot=0,
        refGrid=ref_grid,
        SampleRate=sample_rate
    )
    
    if verbose:
        print(f"  Timing offset: {timing_offset} samples")
    
    return timing_offset
