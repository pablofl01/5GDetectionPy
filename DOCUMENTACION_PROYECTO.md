# 📚 Documentación Completa del Proyecto 5G Detection Python

## 🎯 Objetivo del Proyecto

Este proyecto implementa un **sistema completo de detección y análisis de señales 5G NR** usando Python puro (sin MATLAB). Permite:
- Capturar señales 5G con hardware USRP B210
- Procesar archivos `.mat` con señales capturadas
- Detectar Cell ID (0-1007)
- Visualizar resource grids
- Analizar SSB (Synchronization Signal Blocks)

---

## 📁 Estructura de Archivos del Proyecto

```
5GDetectionPy-1/
├── 📄 demodulate_5g_nr.py          ⭐ SCRIPT PRINCIPAL: Demodulador offline
├── 📄 monitoreo_continuo.py        🔴 Monitoreo en tiempo real con USRP
├── 📄 config.yaml                  ⚙️ Configuración de parámetros
├── 📄 requirements.txt             📦 Dependencias Python
├── 📄 README.md                    📖 Documentación general
├── 📄 USAGE.md                     📘 Manual de uso detallado
├── 📄 CAMBIOS_TIMING.md            📝 Historial de mejoras (timing)
├── 📄 nrTimingEstimate.m           🔍 Referencia MATLAB (timing estimation)
├── 📂 capturas_disco_con/          💾 Capturas con detección
├── 📂 capturas_disco_sin/          💾 Capturas sin detección
├── 📂 resource_grids_output/       🖼️ Imágenes generadas
└── 📂 .venv/                       🐍 Entorno virtual Python
```

---

## 🔧 Archivos Principales

### 1️⃣ `demodulate_5g_nr.py` - **Demodulador Offline** ⭐

**Propósito**: Procesar archivos `.mat` con señales 5G capturadas y extraer información.

**Funcionalidad**:
- Lee archivos `.mat` (v7 o v7.3/HDF5)
- Detecta Cell ID mediante PSS/SSS
- Corrige frecuencia y timing
- Demodula OFDM
- Genera visualizaciones del resource grid
- Guarda logs con resultados

**Funciones principales**:

```python
# 1. Cargar señal desde archivo
load_mat_file(filename: str) -> np.ndarray
    """Carga waveform desde .mat (compatible v7 y v7.3 HDF5)"""

# 2. Corrección de frecuencia + Detección PSS
hssb_burst_frequency_correct_ofdm(waveform, scs, sample_rate, search_bw)
    """
    - Prueba múltiples offsets de frecuencia (-45 kHz a +45 kHz)
    - Genera señales PSS de referencia (NID2=0,1,2)
    - Modula con OFDM usando nrOFDMModulate
    - Correlaciona waveforms
    - Detecta mejor NID2 y frecuencia
    """
    Returns: (waveform_corrected, freq_offset, nid2)

# 3. Estimación de timing offset
estimate_timing_offset(waveform, nid2, scs, sample_rate) -> int
    """
    - Usa nrTimingEstimate de py3gpp
    - Crea reference grid con PSS
    - Encuentra inicio exacto del SSB
    """
    Returns: timing_offset (muestras)

# 4. Detección de Cell ID (SSS)
detect_cell_id_sss(ssb_grid, nid2) -> (nid1, max_corr)
    """
    - Extrae SSS del grid demodulado
    - Prueba 336 valores de NID1
    - Correlaciona con secuencias SSS
    - Formula: sum(abs(sssRx * conj(sssRef))^2)
    """
    Returns: (nid1, correlación_máxima)

# 5. Detectar SSB más fuerte
detect_strongest_ssb(ssb_grids, nid2, nid1, lmax) -> (ssb_idx, power, snr)
    """
    - Analiza Lmax=8 SSB bursts
    - Calcula potencia usando SSS
    - Estima SNR con PBCH-DMRS
    """
    Returns: (strongest_ssb, power_db, snr_db)

# 6. Función principal
demodulate_single(mat_file, scs, gscn, lmax, output_folder)
    """Orquesta todo el proceso de demodulación"""
    Returns: dict con resultados o None si falla
```

**Flujo de ejecución**:
```
1. Cargar waveform desde .mat
2. Corrección de frecuencia → detecta NID2
3. Estimar timing offset
4. Alinear señal (recortar desde timing_offset)
5. Demodular OFDM → obtener resource grid
6. Detectar NID1 usando SSS → Cell ID = 3*NID1 + NID2
7. Analizar 8 SSB bursts → encontrar el más fuerte
8. Generar resource grid para visualización
9. Guardar imagen PNG + log TXT
10. Mostrar resultados
```

**Uso**:
```bash
# Básico (solo muestra resultados)
.venv/bin/python demodulate_5g_nr.py capturas_disco_con/archivo.mat

# Con salida de imágenes
.venv/bin/python demodulate_5g_nr.py capturas_disco_con/archivo.mat resource_grids_output
```

---

### 2️⃣ `monitoreo_continuo.py` - **Monitoreo en Tiempo Real** 🔴

**Propósito**: Capturar y analizar señales 5G en tiempo real usando USRP B210.

**Funcionalidad**:
- Configura y controla USRP B210
- Captura continua de señales IQ
- Procesa señales en tiempo real
- Visualización interactiva con matplotlib
- Guarda capturas en archivos `.mat`
- Modo simulación (sin hardware)

**Secciones principales**:

```python
# 1. Configuración (YAML + CLI)
load_config(config_file) -> dict
merge_config(config, args) -> dict
    """Carga y fusiona configuración desde YAML y línea de comandos"""

# 2. Conversión GSCN ↔ Frecuencia
gscn_to_frequency(gscn: int) -> float
frequency_to_gscn(freq: float) -> int
    """Convierte entre GSCN y frecuencia en Hz"""

# 3. Funciones 5G NR Core
nrPSS(nid2: int) -> np.ndarray
    """Genera secuencia PSS (Primary Sync Signal)"""

nrSSS(ncellid: int) -> np.ndarray
    """Genera secuencia SSS (Secondary Sync Signal)"""

nrPSSIndices() -> np.ndarray
nrSSSIndices() -> np.ndarray
    """Índices de subportadoras donde van PSS/SSS"""

# 4. Corrección de frecuencia
hssb_burst_frequency_correct(waveform, scs_hz, search_bw_hz)
    """
    - Busca offset de frecuencia óptimo
    - Correlaciona con PSS (NID2=0,1,2)
    - Similar a demodulate_5g_nr.py pero más rápido
    """
    Returns: (waveform_corrected, freq_offset, nid2)

# 5. Timing estimation
_timing_estimate(waveform, nid2, nrb_ssb, scs_khz, sample_rate)
    """
    - Replica nrTimingEstimate de MATLAB
    - Modula PSS con OFDM
    - Correlación cruzada
    """
    Returns: timing_offset

# 6. Detección SSB completa
find_ssb(waveform, scs_khz, sample_rate) -> SSBInfo
    """
    Pipeline completo:
    1. Frequency correction
    2. Timing estimation
    3. OFDM demodulation
    4. Cell ID detection (SSS)
    5. PBCH DM-RS analysis
    """
    Returns: dataclass SSBInfo con todos los resultados

# 7. Demodulación OFDM
ofdm_demodulate_ssb(waveform, nrb, scs_khz, nfft, sample_rate)
    """
    - Implementa OFDM demodulation
    - Elimina cyclic prefix
    - FFT de cada símbolo
    """
    Returns: resource_grid (subcarriers × symbols)

# 8. Control USRP
capture_usrp(usrp, num_samples, num_captures, gain_db)
    """Captura señales IQ con USRP"""

setup_usrp(device_args, sample_rate, center_freq, gain, antenna)
    """Configura parámetros del USRP"""

# 9. Simulación
simulate_5g_signal(num_samples, cell_id, snr_db, freq_offset_hz)
    """Genera señal 5G simulada para testing sin hardware"""

# 10. Visualización
plot_resource_grid_interactive(grid, ssb_info, title)
    """
    - Muestra resource grid con matplotlib
    - Marca SSB con rectángulo
    - Permite ajustar contraste con slider
    """

# 11. Guardado de capturas
save_capture_to_mat(waveform, ssb_info, filename)
    """Guarda waveform + metadata en .mat"""
```

**Flujo de ejecución**:
```
1. Cargar configuración (YAML + CLI)
2. Configurar USRP o modo simulación
3. LOOP continuo:
   a. Capturar frames_per_capture frames
   b. Detectar SSB (frequency + timing + Cell ID)
   c. Demodular OFDM → resource grid
   d. Visualizar (GUI interactivo)
   e. Guardar captura .mat (opcional)
   f. Esperar intervalo
   g. Repetir
4. Cerrar USRP
```

**Uso**:
```bash
# Con archivo de configuración
.venv/bin/python monitoreo_continuo.py --config config.yaml

# Con parámetros CLI (sobrescriben config.yaml)
.venv/bin/python monitoreo_continuo.py --gscn 7929 --gain 50 --monitor-time 1.0

# Modo simulación (sin hardware)
.venv/bin/python monitoreo_continuo.py --simulate --no-gui

# Sin GUI (solo logs)
.venv/bin/python monitoreo_continuo.py --no-gui
```

---

### 3️⃣ `config.yaml` - **Archivo de Configuración** ⚙️

**Propósito**: Centralizar todos los parámetros configurables del sistema.

**Secciones**:

```yaml
# DISPOSITIVO USRP
device:
  index: null           # Índice del USRP (0, 1, 2, ...)
  serial: null          # Número de serie
  args: ""              # Argumentos adicionales

# PARÁMETROS RF
rf:
  gscn: 7929            # Canal 5G (GSCN 7929 = 3619.2 MHz)
  sample_rate: 19500000 # 19.5 MHz
  gain: 50              # Ganancia en dB (0-76)
  scs: 30               # Subcarrier spacing (kHz)
  antenna: "RX2"        # Antena a usar

# PROCESADO 5G
processing:
  nrb_ssb: 20           # Resource blocks del SSB (siempre 20)
  nrb_demod: 45         # RBs para demodulación completa
  n_symbols_display: 54 # Símbolos a visualizar
  search_bandwidth_khz: 90  # Búsqueda de frecuencia
  lmax: 8               # Número de SSB bursts

# MONITOREO
monitoring:
  monitor_time: 0.57    # Tiempo total de monitoreo (segundos)
  interval: 0.057       # Intervalo entre capturas
  frames_per_capture: 1 # Frames por captura

# VISUALIZACIÓN
visualization:
  enable_gui: true      # Mostrar ventana interactiva
  verbose: false        # Logs detallados
  save_figures: false   # Guardar PNGs
  contrast_low: 0       # Contraste mínimo
  contrast_high: 50     # Contraste máximo

# SIMULACIÓN
simulation:
  enabled: false        # Usar señal simulada
  cell_id: 0            # Cell ID para simular
  snr_db: 10            # SNR de la simulación
  freq_offset_hz: -2000 # Offset de frecuencia

# EXPORTACIÓN
export:
  enabled: false        # Guardar capturas
  format: "mat"         # Formato: "mat", "hdf5", "npy"
  folder: "capturas"    # Carpeta de salida
```

**Prioridad de configuración**:
```
CLI arguments > config.yaml > valores por defecto
```

---

### 4️⃣ `requirements.txt` - **Dependencias** 📦

```
numpy>=1.24.0,<2.0     # Arrays, matemáticas (¡NOTA: UHD requiere v1.x!)
scipy>=1.7.0           # Señales, FFT, correlación
matplotlib>=3.4.0      # Visualización
pyyaml>=5.4.0          # Leer config.yaml
h5py>=3.0.0            # Leer .mat v7.3 (HDF5)
py3gpp                 # Funciones 5G NR (PSS, SSS, OFDM, etc.)

# uhd: NO está en PyPI, se instala del sistema:
#   sudo apt install python3-uhd uhd-host libuhd-dev
#   ln -s /usr/lib/python3/dist-packages/uhd .venv/.../site-packages/uhd
```

---

### 5️⃣ Archivos de Documentación 📖

#### `README.md`
- Introducción al proyecto
- Características principales
- Instalación rápida
- Ejemplos de uso básico
- Uso programático

#### `USAGE.md`
- Manual detallado de `monitoreo_continuo.py`
- Instalación de UHD
- Configuración completa
- Ejemplos avanzados
- Troubleshooting

#### `CAMBIOS_TIMING.md`
- Historia de mejoras en timing estimation
- Explicación del problema original
- Implementación de `nrTimingEstimate`
- Proceso correcto de recorte de señal
- Comparación con MATLAB

#### `nrTimingEstimate.m`
- Código MATLAB de referencia
- Documentación de la función
- Usado como guía para implementación Python

---

## 🔄 Flujo de Datos en `demodulate_5g_nr.py`

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CARGA DE SEÑAL                                           │
├─────────────────────────────────────────────────────────────┤
│ load_mat_file(archivo.mat)                                  │
│   → waveform: array complejo [390000 muestras]             │
│   → Sample rate: 19.5 MHz                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. CORRECCIÓN DE FRECUENCIA + DETECCIÓN PSS                │
├─────────────────────────────────────────────────────────────┤
│ hssb_burst_frequency_correct_ofdm()                         │
│   • Prueba offsets: -45 kHz a +45 kHz (65 valores)         │
│   • Para cada offset y NID2 (0,1,2):                        │
│     - Aplica corrección: wf * exp(-j*2*pi*f*t)             │
│     - Genera PSS de referencia                              │
│     - Modula con OFDM: nrOFDMModulate(ref_grid)            │
│     - Correlaciona: correlate(wf_corrected, ref_waveform)   │
│   • Encuentra máximo global → (freq_offset, NID2)          │
│                                                              │
│   → waveform_corrected: señal con frecuencia corregida     │
│   → freq_offset: -2.0 kHz                                   │
│   → nid2: 0                                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. ESTIMACIÓN DE TIMING OFFSET                             │
├─────────────────────────────────────────────────────────────┤
│ estimate_timing_offset()                                    │
│   • Crea reference grid con PSS en símbolo 1               │
│   • Usa nrTimingEstimate() de py3gpp                       │
│   • Correlación para encontrar inicio exacto del SSB       │
│                                                              │
│   → timing_offset: 215895 muestras                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. ALINEAMIENTO DE SEÑAL (¡CLAVE!)                         │
├─────────────────────────────────────────────────────────────┤
│ waveform_aligned = waveform_corrected[timing_offset:]      │
│                                                              │
│ • Recorta la señal desde el inicio del SSB                 │
│ • Ahora SSB empieza en la muestra 0                        │
│ • ¡Esto garantiza que SSB aparezca al inicio del grid!     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. DEMODULACIÓN OFDM                                        │
├─────────────────────────────────────────────────────────────┤
│ nrOFDMDemodulate(waveform_aligned, nrb=20, scs=30)         │
│   • Para cada símbolo OFDM:                                 │
│     - Quitar cyclic prefix                                  │
│     - FFT de 256 puntos                                     │
│     - Extraer 240 subportadoras (20 RBs)                   │
│   • Genera grid: [240 subportadoras × 4 símbolos]         │
│                                                              │
│   → grid_ssb: resource grid del SSB [240, 4]               │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. DETECCIÓN DE CELL ID (SSS)                              │
├─────────────────────────────────────────────────────────────┤
│ detect_cell_id_sss(grid_ssb, nid2)                         │
│   • Extrae SSS del grid (símbolo 2)                        │
│   • Para NID1 = 0 a 335:                                   │
│     - cell_id = 3*NID1 + NID2                              │
│     - sss_ref = nrSSS(cell_id)                             │
│     - correlation = sum(abs(sss_rx * conj(sss_ref))^2)     │
│   • Encuentra NID1 con máxima correlación                  │
│                                                              │
│   → nid1: 0                                                 │
│   → cell_id: 3*0 + 0 = 0                                   │
│   → max_corr: 3.84                                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. ANÁLISIS DE MÚLTIPLES SSB BURSTS                        │
├─────────────────────────────────────────────────────────────┤
│ Para i_ssb = 0 a 7 (Lmax=8):                               │
│   • Extraer porción de señal (periodicidad 20ms/8)         │
│   • Demodular OFDM                                          │
│   • Guardar en ssb_grids[:, :, i_ssb]                      │
│                                                              │
│ detect_strongest_ssb(ssb_grids, nid2, nid1, lmax=8)        │
│   • Calcular potencia de cada SSB (usando SSS)             │
│   • Estimar SNR (usando PBCH-DMRS)                         │
│   • Encontrar SSB con mayor potencia                       │
│                                                              │
│   → strongest_ssb: 2                                        │
│   → power_db: -14.0 dB                                     │
│   → snr_db: -6.9 dB                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. CREACIÓN DE RESOURCE GRID PARA VISUALIZACIÓN            │
├─────────────────────────────────────────────────────────────┤
│ nrOFDMDemodulate(waveform_aligned, nrb=45, ...)            │
│   • Demodula grid más amplio (45 RBs = 540 subportadoras) │
│   • Toma primeros 54 símbolos OFDM                         │
│   • SSB (20 RBs) aparece centrado en frecuencia            │
│   • SSB aparece en símbolos 0-3 (izquierda en tiempo)     │
│                                                              │
│   → grid_display: [540 subportadoras × 54 símbolos]       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 9. GUARDADO DE RESULTADOS                                  │
├─────────────────────────────────────────────────────────────┤
│ • Imagen PNG: resource grid con colormap 'jet'             │
│ • Log TXT: Cell ID, SNR, offsets, timestamps               │
│ • Consola: Resumen de resultados                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Conceptos Clave de 5G NR

### Cell ID
```
Cell ID = 3 × NID1 + NID2

NID1: 0-335 (336 valores) → Physical Cell ID Group
NID2: 0-2 (3 valores)     → Physical Layer Identity

Rango total: 0-1007 (1008 Cell IDs posibles)
```

### SSB (Synchronization Signal Block)
```
SSB = PSS + SSS + PBCH + PBCH-DMRS

Estructura temporal (4 símbolos OFDM):
┌─────────┬─────────┬─────────┬─────────┐
│ Sym 0   │ Sym 1   │ Sym 2   │ Sym 3   │
├─────────┼─────────┼─────────┼─────────┤
│ PSS     │ PBCH    │ SSS +   │ PBCH    │
│         │ + DMRS  │ PBCH    │ + DMRS  │
└─────────┴─────────┴─────────┴─────────┘

Ancho: 20 RBs = 240 subportadoras = 7.2 MHz (para SCS=30kHz)
```

### PSS (Primary Synchronization Signal)
- 3 secuencias (NID2 = 0, 1, 2)
- 127 subportadoras
- Símbolo 0 del SSB
- Usado para detección inicial y timing

### SSS (Secondary Synchronization Signal)
- 336 secuencias (NID1 = 0-335)
- 127 subportadoras
- Símbolo 2 del SSB
- Usado para identificar Cell ID completo

### PBCH (Physical Broadcast Channel)
- Lleva MIB (Master Information Block)
- Símbolos 1, 2, 3 del SSB
- Contiene información del sistema

### GSCN (Global Synchronization Channel Number)
```
Banda n78 (3.3-3.8 GHz):
GSCN = 7499-8255

Ejemplo:
GSCN 7929 → 3619.2 MHz
GSCN 7880 → 3604.56 MHz

Fórmula:
freq_MHz = (N - offset) × step
Para banda n78: step=1.44 MHz, offset variable
```

---

## 🧪 Bibliotecas Clave Usadas

### `py3gpp`
Implementación Python de funciones 5G NR estándar:
```python
from py3gpp.nrPSS import nrPSS
from py3gpp.nrSSS import nrSSS
from py3gpp.nrPSSIndices import nrPSSIndices
from py3gpp.nrSSSIndices import nrSSSIndices
from py3gpp.nrOFDMDemodulate import nrOFDMDemodulate
from py3gpp.nrOFDMModulate import nrOFDMModulate
from py3gpp.nrTimingEstimate import nrTimingEstimate
from py3gpp.nrExtractResources import nrExtractResources
from py3gpp.nrPBCHDMRS import nrPBCHDMRS
from py3gpp.nrPBCHDMRSIndices import nrPBCHDMRSIndices
```

### `uhd` (USRP Hardware Driver)
Control de hardware USRP:
```python
import uhd

# Crear objeto USRP
usrp = uhd.usrp.MultiUSRP(args)

# Configurar
usrp.set_rx_rate(sample_rate)
usrp.set_rx_freq(center_freq)
usrp.set_rx_gain(gain)

# Capturar
samples = usrp.recv_num_samps(num_samples, freq, rate, channels=[0])
```

### NumPy / SciPy
Procesamiento de señales:
```python
# Correlación
scipy.signal.correlate(signal, reference)

# FFT
np.fft.fft(signal)
np.fft.ifft(signal)

# Resample
scipy.signal.resample(signal, num_samples)
```

---

## 🎓 Comparación con MATLAB

| Aspecto | MATLAB | Python (este proyecto) |
|---------|--------|------------------------|
| **Licencia** | Comercial ($$$) | Open Source (GRATIS) |
| **Toolboxes** | 5G Toolbox | py3gpp |
| **PSS/SSS** | `nrPSS()`, `nrSSS()` | `py3gpp.nrPSS()`, `py3gpp.nrSSS()` |
| **OFDM** | `nrOFDMModulate()` | `py3gpp.nrOFDMModulate()` |
| **Timing** | `nrTimingEstimate()` | `py3gpp.nrTimingEstimate()` |
| **Hardware** | Communications Toolbox | python3-uhd |
| **Visualización** | MATLAB plots | matplotlib |
| **Performance** | Optimizado | Comparable (NumPy) |

**Ventajas de Python**:
- ✅ Totalmente gratuito
- ✅ Open source (auditable)
- ✅ Integración con ecosistema científico (NumPy, SciPy, ML)
- ✅ Fácil despliegue en servidores Linux
- ✅ Gran comunidad y recursos

---

## 🚀 Casos de Uso

### 1. Análisis Offline (Archivo .mat)
```bash
# Analizar una captura guardada
.venv/bin/python demodulate_5g_nr.py capturas_disco_con/archivo.mat output/
```
**Resultado**: Cell ID, SNR, resource grid PNG

### 2. Monitoreo en Tiempo Real
```bash
# Capturar y analizar continuamente
.venv/bin/python monitoreo_continuo.py --config config.yaml
```
**Resultado**: Visualización interactiva + capturas guardadas

### 3. Testing sin Hardware
```bash
# Simular señal 5G
.venv/bin/python monitoreo_continuo.py --simulate --no-gui
```
**Resultado**: Validar algoritmos sin USRP

### 4. Batch Processing
```python
from pathlib import Path
from demodulate_5g_nr import demodulate_single

# Procesar todas las capturas
for mat_file in Path('capturas/').glob('*.mat'):
    result = demodulate_single(str(mat_file), output_folder='resultados/')
    if result:
        print(f"✓ {mat_file.name}: Cell ID={result['cell_id']}, SNR={result['snr_db']:.1f} dB")
```

### 5. Integración en Sistema Mayor
```python
from monitoreo_continuo import find_ssb, setup_usrp, capture_usrp

# Configurar USRP
usrp = setup_usrp("", 19.5e6, 3619.2e6, 50, "RX2")

# Capturar
frames = capture_usrp(usrp, 390000, 1, 50)
waveform = frames[0]

# Detectar SSB
ssb_info = find_ssb(waveform, scs_khz=30, sample_rate=19.5e6)

if ssb_info and ssb_info.detected:
    print(f"Cell ID: {ssb_info.cell_id}, SNR: {ssb_info.snr_db:.1f} dB")
```

---

## 🔍 Troubleshooting Común

### Problema: `ModuleNotFoundError: No module named 'py3gpp'`
**Solución**: Instalar py3gpp en el entorno virtual
```bash
source .venv/bin/activate
pip install py3gpp
```

### Problema: `ModuleNotFoundError: No module named 'uhd'`
**Solución**: Enlazar UHD del sistema al virtualenv
```bash
sudo apt install python3-uhd uhd-host
ln -s /usr/lib/python3/dist-packages/uhd .venv/lib/python3.*/site-packages/uhd
```

### Problema: SSB no se detecta (NID1 inválido)
**Causas**:
- Señal muy débil (aumentar `gain`)
- Frecuencia incorrecta (verificar `gscn`)
- Interferencia (cambiar ubicación/antena)

### Problema: Resource grid vacío o ruidoso
**Causas**:
- Timing offset incorrecto (verificar implementación)
- Sample rate incompatible
- Señal demasiado débil

### Problema: Error en `nrOFDMDemodulate`
**Solución**: Verificar que waveform tenga suficientes muestras
```python
min_samples = (nfft + max_cp_length) * num_symbols_needed
if len(waveform) < min_samples:
    # Señal muy corta
```

---

## 📈 Próximas Mejoras

- [ ] Decodificación completa de PBCH (MIB)
- [ ] Detección de PDCCH/PDSCH
- [ ] Estimación de canal MIMO
- [ ] Soporte para más bandas (n1, n3, n7, n41, etc.)
- [ ] Dashboard web (Flask/FastAPI)
- [ ] Base de datos de capturas (PostgreSQL)
- [ ] Machine Learning para clasificación de señales
- [ ] Procesamiento distribuido (múltiples USRPs)

---

## 📞 Contacto y Contribuciones

**Repositorio**: github.com/pablofl01/5GDetectionPy
**Branch**: master

Para contribuir:
1. Fork del proyecto
2. Crear branch con feature
3. Pull request con descripción detallada

---

**Fecha de esta documentación**: 16 de diciembre de 2025
**Versión del proyecto**: 1.0
**Autor**: [Tu nombre/equipo]
