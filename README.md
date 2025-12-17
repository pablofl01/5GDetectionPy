# 5G NR Demodulator

5G NR signal demodulator developed in Python that detects Cell ID, SSB, power and SNR from `.mat` files captured with SDR or in real-time with USRP B210.

## Installation

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For USRP capture (optional)
pip install uhd pyyaml
```

## Configuration

Default parameters are defined in `config.yaml`. This file centralizes all RF, processing, visualization and export parameters. CLI arguments override values from `config.yaml`.

### Main parameters in config.yaml

```yaml
rf:
  gscn: 7929                 # Global Sync Channel Number
  sample_rate: 19.5e6       # Sample rate (Hz)
  scs: 30                   # Subcarrier spacing (kHz)

processing:
  nrb_ssb: 20               # Resource blocks for SSB
  nrb_demod: 45             # Resource blocks for demodulation
  search_bw: 90             # Search bandwidth (kHz)
```

## Usage

### 1. Processing .mat files (CLI)

```bash
# Process file (uses config.yaml)
python demodulate_cli.py capture.mat -o results/

# Override config.yaml parameters
python demodulate_cli.py capture.mat -o results/ --scs 15 --gscn 7880

# Process entire folder
python demodulate_cli.py folder/ -o results/ --pattern "*.mat"

# View options
python demodulate_cli.py --help
```

### 2. Programmatic usage (API)

```python
from nr_demodulator import demodulate_file, demodulate_ssb

# Process .mat file
result = demodulate_file('capture.mat', output_dir='results/', scs=30)
print(f"Cell ID: {result['cell_id']}, SNR: {result['snr_db']:.1f} dB")

# Demodulate waveform in memory (live capture)
waveform = ...  # complex numpy array from SDR
result = demodulate_ssb(waveform, scs=30, sample_rate=19.5e6, lmax=8)
```

### 3. Simple capture with USRP B210

Simplified script for quick capture, demodulation and visualization with axes:

```bash
# Simple capture (uses config.yaml)
python captura_simple.py

# With specific GSCN
python captura_simple.py --gscn 7880

# List USRP devices
python captura_simple.py --list-devices

# Select device
python captura_simple.py --device-index 0
python captura_simple.py --device-serial 12345678

# Adjust gain and duration
python captura_simple.py --gain 40 --duration 0.05
```

**Features**:
- Single capture (doesn't save files)
- Visualization with axes (X: OFDM symbols, Y: subcarriers)
- Moderate logs (not verbose)
- Ideal for quick testing

### 4. Continuous monitoring with USRP B210

For multiple captures and prolonged monitoring:

```bash
# List USRP devices
python monitoreo_continuo.py --list-devices

# Continuous capture and processing
python monitoreo_continuo.py --config config.yaml

# Simulation mode (no hardware)
python monitoreo_continuo.py --simulate
```

**Features**:
- Multiple captures with configurable interval
- Visualization with temporal slider
- Optional result saving
- Complete parameter control

## Project structure

```
5GDetectionPy/
├── nr_demodulator.py           # Main demodulation API
├── frequency_correction.py     # Frequency offset correction
├── timing_estimation.py        # Timing offset estimation
├── cell_detection.py           # Cell ID and SSB detection
├── visualization.py            # Visualization and logging
├── config_loader.py            # YAML configuration loader
├── demodulate_cli.py           # CLI for .mat files
├── captura_simple.py           # Quick capture with USRP
├── monitoreo_continuo.py       # Continuous USRP monitoring
└── config.yaml                 # Centralized configuration
```

## Output

When processing files, the following are generated:

- `<file>_resource_grid.png` - Resource grid visualization (540×54, 300 DPI)
- `<file>_info.txt` - Log with Cell ID, NID1/NID2, SNR, power, offsets
- `<file>_ERROR.txt` - Error log (if errors occur)

## Parameters

All scripts use `config.yaml` by default. CLI arguments override these values.

### demodulate_cli.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--scs` | Subcarrier spacing (kHz) | config.yaml (30) |
| `--gscn` | Channel GSCN | config.yaml (7929) |
| `--lmax` | Number of SSB bursts | 8 |
| `--pattern` | File pattern | `*.mat` |
| `--verbose` | Detailed mode | False |
| `--show-axes` | Images with axes | False |
| `--no-plot` | Don't save images | False |

### captura_simple.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--device-index` | USRP device index | Auto |
| `--device-serial` | USRP device serial | Auto |
| `--gscn` | Channel GSCN | config.yaml (7929) |
| `--scs` | Subcarrier spacing (kHz) | config.yaml (30) |
| `--gain` | Receiver gain (dB) | config.yaml (50) |
| `--duration` | Capture duration (s) | 0.02 |
| `--list-devices` | List USRP devices | - |

## Troubleshooting

```bash
# Missing modules
pip install py3gpp h5py uhd pyyaml

# Verify USRP
uhd_find_devices
python monitoreo_continuo.py --list-devices

# Without USRP hardware
python monitoreo_continuo.py --simulate
```

## References

- [py3gpp](https://github.com/NajibOdhah/py3gpp) - Python implementation of 5G NR
- [3GPP TS 38.211](https://www.3gpp.org/DynaReport/38211.htm) - Physical channels and modulation
