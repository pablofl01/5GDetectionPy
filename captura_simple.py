#!/usr/bin/env python3
"""
Simple 5G NR capture and demodulation script with USRP B210.
Captures a signal, demodulates it and displays the resource grid with axes.
"""

import uhd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

from config_loader import get_config
from nr_demodulator import demodulate_ssb
from visualization import plot_resource_grid


def list_usrp_devices():
    """Lists available USRP devices."""
    print('\n=== AVAILABLE USRP DEVICES ===')
    device_addrs = uhd.find("")

    if not device_addrs:
        print('No USRP devices found connected.')
        return []

    devices = []
    for idx, addr in enumerate(device_addrs):
        print(f'\n[{idx}] Device found:')
        device_info = {}
        for key in addr.keys():
            value = addr.get(key)
            print(f'    {key}: {value}')
            device_info[key] = value
        devices.append(device_info)

    print('\n' + '=' * 40)
    return devices


def select_usrp_device(device_index=None, device_serial=None):
    """Selects a USRP device."""
    devices = list_usrp_devices()

    if not devices:
        raise RuntimeError("No USRP devices available")

    if device_index is not None:
        if 0 <= device_index < len(devices):
            selected = devices[device_index]
            print(f'\n✓ Selected device [{device_index}]: {selected.get("serial", "N/A")}')
            if 'serial' in selected:
                return f"serial={selected['serial']}"
            return ""
        else:
            raise ValueError(f"Index {device_index} out of range. There are {len(devices)} devices.")

    if device_serial is not None:
        for dev in devices:
            if dev.get('serial') == device_serial:
                print(f'\n✓ Selected device with serial: {device_serial}')
                return f"serial={device_serial}"
        raise ValueError(f"Device with serial not found: {device_serial}")

    if len(devices) == 1:
        selected = devices[0]
        print(f'\n✓ Using only available device: {selected.get("serial", "N/A")}')
        if 'serial' in selected:
            return f"serial={selected['serial']}"
        return ""

    print(f'\n⚠ There are {len(devices)} devices. Specify --device-index or --device-serial')
    raise RuntimeError("Multiple devices found. Specify which one to use.")


def gscn_to_frequency(gscn: int) -> float:
    """Converts GSCN to frequency in Hz."""
    if 7499 <= gscn <= 22255:
        N = gscn - 7499
        freq_hz = 3000e6 + N * 1.44e6
        return freq_hz
    else:
        raise ValueError(f"GSCN {gscn} out of FR1 range")


def capture_waveform(center_freq, sample_rate, gain, duration, device_args=""):
    """Captures a signal with the USRP B210."""
    print('\n--- Configuring USRP B210 ---')
    
    # Create USRP object
    usrp = uhd.usrp.MultiUSRP(device_args)
    
    # Configure sample rate
    usrp.set_rx_rate(sample_rate, 0)
    actual_rate = usrp.get_rx_rate(0)
    print(f'Sample rate: {actual_rate/1e6:.2f} MHz')
    
    # Configure center frequency
    tune_request = uhd.types.TuneRequest(center_freq)
    usrp.set_rx_freq(tune_request, 0)
    actual_freq = usrp.get_rx_freq(0)
    print(f'Center frequency: {actual_freq/1e6:.2f} MHz')
    
    # Configure gain
    usrp.set_rx_gain(gain, 0)
    actual_gain = usrp.get_rx_gain(0)
    print(f'Gain: {actual_gain:.1f} dB')
    
    # Configure antenna
    usrp.set_rx_antenna("RX2", 0)
    print(f'Antenna: {usrp.get_rx_antenna(0)}')
    
    # Capture
    print(f'\n--- Capturing {duration*1000:.1f} ms ---')
    num_samples = int(duration * sample_rate)
    samples = np.zeros(num_samples, dtype=np.complex64)
    
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    stream_args.channels = [0]
    rx_streamer = usrp.get_rx_stream(stream_args)
    
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
            print(f'⚠ Reception error: {metadata.strerror()}')
            break
        end_idx = min(samples_received + num_rx_samps, num_samples)
        samples[samples_received:end_idx] = recv_buffer[0, :end_idx - samples_received]
        samples_received = end_idx
    
    print(f'✓ Captured {len(samples)} samples')
    power_dbm = 10 * np.log10(np.mean(np.abs(samples)**2) + 1e-12)
    print(f'✓ Signal power: {power_dbm:.1f} dB')
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Simple 5G NR signal capture with USRP B210',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Usage examples:
  %(prog)s
  %(prog)s --gscn 7880
  %(prog)s --device-index 0
  %(prog)s --gain 40
  %(prog)s --list-devices
        '''
    )
    
    parser.add_argument('--list-devices', action='store_true',
                        help='List available USRP devices and exit')
    parser.add_argument('--device-index', type=int, metavar='N',
                        help='Device index to use (0, 1, 2, ...)')
    parser.add_argument('--device-serial', type=str, metavar='SERIAL',
                        help='Device serial number to use')
    parser.add_argument('--gscn', type=int,
                        help='Channel GSCN (default: from config.yaml)')
    parser.add_argument('--scs', type=int, choices=[15, 30],
                        help='Subcarrier spacing in kHz (default: from config.yaml)')
    parser.add_argument('--gain', type=float,
                        help='Receiver gain in dB (default: from config.yaml)')
    parser.add_argument('--duration', type=float, default=0.02,
                        help='Capture duration in seconds (default: 0.02)')
    
    args = parser.parse_args()
    
    # Listar dispositivos si se solicita
    if args.list_devices:
        list_usrp_devices()
        return
    
    # Cargar configuración
    config = get_config()
    
    # Usar valores de config o argumentos CLI
    gscn = args.gscn if args.gscn is not None else config.gscn
    scs = args.scs if args.scs is not None else config.scs
    gain = args.gain if args.gain is not None else config.gain
    sample_rate = config.sample_rate
    
    print('=== 5G NR CAPTURE AND DEMODULATION ===\n')
    print(f'Configuration:')
    print(f'  GSCN: {gscn}')
    print(f'  SCS: {scs} kHz')
    print(f'  Gain: {gain} dB')
    print(f'  Sample rate: {sample_rate/1e6:.2f} MHz')
    print(f'  Duration: {args.duration*1000:.1f} ms')
    
    # Calculate center frequency
    center_freq = gscn_to_frequency(gscn)
    print(f'  Frequency: {center_freq/1e6:.2f} MHz')
    
    try:
        # Select device
        device_args = select_usrp_device(
            device_index=args.device_index,
            device_serial=args.device_serial
        )
        
        # Capture signal
        waveform = capture_waveform(
            center_freq=center_freq,
            sample_rate=sample_rate,
            gain=gain,
            duration=args.duration,
            device_args=device_args
        )
        
        # Demodulate
        print('\n--- Demodulating ---')
        n_symbols = config.n_symbols_display
        results = demodulate_ssb(waveform, scs=scs, sample_rate=sample_rate, 
                                n_symbols_display=n_symbols, verbose=False)
        
        # Display results
        print('\n=== RESULTS ===')
        print(f'Cell ID: {results["cell_id"]}')
        print(f'  NID1: {results["nid1"]}')
        print(f'  NID2: {results["nid2"]}')
        print(f'Strongest SSB: {results["strongest_ssb"]}')
        print(f'Power: {results["power_db"]:.1f} dB')
        print(f'SNR: {results["snr_db"]:.1f} dB')
        print(f'Freq offset: {results["freq_offset"]/1e3:.3f} kHz')
        print(f'Timing offset: {results["timing_offset"]} samples')
        
        # Visualize resource grid with axes
        print('\n--- Displaying Resource Grid ---')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        grid = results['grid_display']
        im = ax.imshow(np.abs(grid), aspect='auto', cmap='jet',
                      origin='lower', interpolation='nearest')
        
        plt.colorbar(im, ax=ax, label='Magnitude')
        ax.set_xlabel('OFDM Symbol')
        ax.set_ylabel('Subcarrier')
        ax.set_title(f'Resource Grid - Cell ID: {results["cell_id"]}, '
                    f'SNR: {results["snr_db"]:.1f} dB '
                    f'({center_freq/1e6:.2f} MHz)')
        
        plt.tight_layout()
        plt.show()
        
        print('\n✓ Process completed')
        
    except KeyboardInterrupt:
        print('\n\n⚠ Interrupted by user')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ Error: {e}')
        print('\nYou can use --list-devices to see available devices')
        sys.exit(1)


if __name__ == '__main__':
    main()
