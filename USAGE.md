# Monitoreo Continuo de Señales 5G NR con USRP B210

Script Python para monitoreo y análisis de señales 5G NR usando USRP B210. Equivalente al script MATLAB `MonitoreoContinuoFunciones.m`.

## 🎯 Características

- ✅ Captura continua de señales 5G NR
- ✅ Detección automática de SSB (Synchronization Signal Block)
- ✅ Corrección automática de frecuencia y timing
- ✅ Visualización interactiva de resource grids
- ✅ Identificación de Cell ID (0-1007)
- ✅ Soporte multi-dispositivo USRP
- ✅ Modo simulación sin hardware
- ✅ **Sistema de configuración flexible (YAML + CLI)**

## 📋 Requisitos

### Hardware
- USRP B210 (u otro modelo compatible con UHD)
- Antena para banda 5G NR (ej: banda n78, 3.3-3.8 GHz)

### Software
- Python 3.8+
- UHD (USRP Hardware Driver) 4.x
- NumPy < 2.0 (requerido por UHD)
- SciPy, Matplotlib, PyYAML

## 🚀 Instalación Rápida

```bash
# 1. Instalar UHD (sistema)
sudo apt update
sudo apt install python3-uhd uhd-host libuhd-dev

# 2. Crear entorno virtual
python3 -m venv env
source env/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Enlazar UHD al virtualenv
ln -s /usr/lib/python3/dist-packages/uhd env/lib/python3.12/site-packages/uhd

# 5. Probar en modo simulación
python monitoreo_continuo.py --simulate --no-gui
```

## ⚙️ Configuración

### Opción 1: Archivo YAML (Recomendado)

Edita `config.yaml`:

```yaml
rf:
  gscn: 7929              # Frecuencia (GSCN)
  sample_rate: 19500000   # 19.5 MHz
  gain: 50                # Ganancia en dB
  scs: 30                 # Subcarrier spacing (kHz)

monitoring:
  monitor_time: 0.57      # Tiempo total (s)
  interval: 0.057         # Intervalo entre capturas (s)
  frames_per_capture: 1   # Frames por captura

visualization:
  enable_gui: true
  verbose: false
```

Ejecutar:

```bash
python monitoreo_continuo.py --config config.yaml
```

### Opción 2: Línea de Comandos

```bash
python monitoreo_continuo.py --gscn 7929 --gain 50 --monitor-time 1.0
```

### Opción 3: Híbrido (CLI sobrescribe YAML)

```bash
# Usar config.yaml pero cambiar GSCN y ganancia
python monitoreo_continuo.py --config config.yaml --gscn 7880 --gain 40
```

**Prioridad**: CLI > Archivo YAML > Valores por defecto

## 📖 Ejemplos de Uso

### Monitoreo Básico

```bash
# Con configuración por defecto
python monitoreo_continuo.py

# Con archivo de configuración
python monitoreo_continuo.py --config config.yaml

# Modo verbose (mostrar detalles de detección)
python monitoreo_continuo.py --config config.yaml --verbose
```

### Listar Dispositivos

```bash
python monitoreo_continuo.py --list-devices
```

### Seleccionar Dispositivo USRP

```bash
# Por índice (0, 1, 2...)
python monitoreo_continuo.py --device-index 0

# Por número de serie
python monitoreo_continuo.py --device-serial 32345E1

# En config.yaml
device:
  serial: "32345E1"
```

### Cambiar Frecuencia

```bash
# GSCN 7880 = 3548.64 MHz
python monitoreo_continuo.py --gscn 7880

# Usar configuración ejemplo
python monitoreo_continuo.py --config config_example_7880.yaml
```

### Ajustar Visualización

```bash
# Mostrar solo 8 símbolos OFDM (más compacto)
python monitoreo_continuo.py --n-symbols-display 8

# Mostrar slot completo (14 símbolos)
python monitoreo_continuo.py --n-symbols-display 14

# En config.yaml
processing:
  n_symbols_display: 10
```

### Ajustar Ganancia

```bash
# Señal débil
python monitoreo_continuo.py --gain 70

# Señal fuerte
python monitoreo_continuo.py --gain 30
```

### Modo Simulación

```bash
# Con GUI
python monitoreo_continuo.py --simulate

# Sin GUI (solo consola)
python monitoreo_continuo.py --simulate --no-gui
```

### Captura Larga

```bash
# 5 segundos, intervalo 0.2s
python monitoreo_continuo.py --monitor-time 5.0 --interval 0.2
```

## 📊 Parámetros Principales

### GSCN (Global Synchronization Channel Number)

Tabla de frecuencias **banda n78**:

| GSCN | Frecuencia (MHz) | Uso |
|------|------------------|-----|
| 7499 | 3000.00 | Inicio banda |
| 7700 | 3289.44 | - |
| 7880 | 3548.64 | Común en Europa |
| **7929** | **3619.20** | **Por defecto** |
| 8100 | 3865.44 | - |
| 8255 | 4088.64 | Fin banda |

**Fórmula**: `freq_MHz = 3000 + (GSCN - 7499) × 1.44`

### Ganancia

Rango USRP B210: **0-76 dB**

| Nivel Señal | Ganancia Recomendada |
|-------------|---------------------|
| Muy débil (rural, lejos) | 65-76 dB |
| Débil | 55-65 dB |
| Media | 40-55 dB |
| Fuerte (cerca antena) | 20-40 dB |
| Muy fuerte | 10-20 dB |

⚠️ **Saturación**: Si `Pot > -5 dB`, reducir ganancia

### Sample Rate

| Valor | Uso |
|-------|-----|
| 15.36 MHz | Mínimo |
| **19.5 MHz** | **Por defecto** (buen balance) |
| 23.04 MHz | Mayor ancho banda |
| 30.72 MHz | Máximo |

### Subcarrier Spacing (SCS)

| SCS | Banda Típica |
|-----|--------------|
| 15 kHz | FR1 baja frecuencia |
| **30 kHz** | **FR1 banda n78** |
| 60 kHz | FR1 alta frecuencia |
| 120 kHz | FR2 (mmWave) |

### Símbolos OFDM a Mostrar

Controla cuántos símbolos OFDM se demodularán y mostrarán en la gráfica.

| Valor | Descripción | Uso |
|-------|-------------|-----|
| 6 | Mínimo | Solo SSB visible (símbolos 2-5 + margen) |
| 8-10 | Compacto | Vista reducida, procesado más rápido |
| **14** | **Slot completo** | **Por defecto, vista completa** |

⚠️ **Nota**: Mínimo 6 símbolos para incluir SSB (símbolos 2-5)

## 🎮 Navegación en Visualización

- **Slider inferior**: Seleccionar frame
- **Flecha →**: Frame siguiente
- **Flecha ←**: Frame anterior
- **Rectángulo rojo**: Posición del SSB (símbolos 2-5)

## 📁 Archivos de Configuración

```
config.yaml                  # Configuración por defecto (GSCN 7929)
config_example_7880.yaml     # Ejemplo GSCN 7880 con verbose
```

### Estructura Completa

```yaml
device:           # Selección de dispositivo USRP
  index: null
  serial: null
  args: ""

rf:               # Parámetros RF
  gscn: 7929
  sample_rate: 19500000
  gain: 50
  scs: 30
  antenna: "RX2"

processing:       # Procesado 5G NR
  nrb_ssb: 20
  nrb_demod: 45
  n_symbols_display: 14  # Símbolos OFDM a mostrar (6-14)
  search_bw: 90
  detection_threshold: 1e-3

monitoring:       # Captura
  monitor_time: 0.57
  interval: 0.057
  frames_per_capture: 1
  save_captures: false

visualization:    # Interfaz
  enable_gui: true
  colormap: "jet"
  verbose: false

simulation:       # Sin hardware
  enabled: false

export:           # Exportar datos
  save_mat: false
  output_dir: "resultados"
```

## 📤 Salida del Script

```
[Demod 1/10] ✓ Tiempo: 0.158s | Pot=-12.3dB | SNR=8.5dB | cellID=267 | Corr=1.34e+05
```

| Campo | Descripción |
|-------|-------------|
| **✓/✗** | SSB detectado/no detectado |
| **Tiempo** | Tiempo de procesado (s) |
| **Pot** | Potencia recibida (dB) |
| **SNR** | Relación señal/ruido (dB) |
| **cellID** | Physical Cell ID (0-1007) |
| **Corr** | Correlación máxima (umbral: 1e-3) |

## 🛠️ Herramientas Adicionales

### compare_grids.py

Diagnóstico avanzado: captura y guarda resource grid en formato `.mat` para comparar con MATLAB.

```bash
# Capturar y visualizar
python compare_grids.py --gscn 7929 --gain 50 --plot

# Luego en MATLAB
data = load('grid_python.mat');
imagesc(abs(data.resourceGrid)); colormap jet; colorbar;
```

## 🔧 Troubleshooting

### Error: ModuleNotFoundError: uhd

```bash
# Verificar instalación
dpkg -l | grep uhd

# Reinstalar
sudo apt install --reinstall python3-uhd

# Recrear enlace
rm -f env/lib/python3.12/site-packages/uhd
ln -s /usr/lib/python3/dist-packages/uhd env/lib/python3.12/site-packages/uhd
```

### Error: Múltiples dispositivos encontrados

```bash
# Ver lista
python monitoreo_continuo.py --list-devices

# Seleccionar
python monitoreo_continuo.py --device-index 0

# O en config.yaml
device:
  index: 0
```

### No se detecta SSB

1. **Verificar GSCN** - Usar escáner de espectro o app celular
2. **Ajustar ganancia** - Empezar con 50 dB, subir si necesario
3. **Verificar antena** - Conexión firme, orientación
4. **Diagnóstico**:
   ```bash
   python compare_grids.py --gscn 7929 --gain 60 --plot
   ```
5. **Modo verbose**:
   ```bash
   python monitoreo_continuo.py --verbose
   ```

### Error al cargar config.yaml

```bash
# Verificar sintaxis YAML
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Reinstalar PyYAML
pip install --force-reinstall pyyaml
```

### SSB no aparece en símbolos 2-5

El script **automáticamente** ajusta el timing offset para posicionar SSB en símbolos 2-5. Si no aparece:

1. Verificar que `verbose: true` en config
2. Revisar mensaje: `"SSB debería aparecer en símbolos 2-5"`
3. Comprobar correlación > 1e-3

## 📊 Diferencias con MATLAB Original

| Aspecto | MATLAB | Python |
|---------|--------|--------|
| **Configuración** | Hardcoded | YAML + CLI |
| **Device selection** | Manual | Auto + multi-device |
| **SSB positioning** | Variable | **Símbolos 2-5 fijos** |
| **Freq correction** | Función específica | Implementado |
| **Timing offset** | nrTimingEstimate | Correlación PSS |
| **Visualización** | Figure callbacks | Matplotlib Slider |
| **Modo simulación** | No | Sí |

## 📂 Estructura del Proyecto

```
5GDetection/
├── monitoreo_continuo.py         # Script principal
├── config.yaml                    # Config por defecto
├── config_example_7880.yaml       # Ejemplo alternativo
├── compare_grids.py               # Herramienta diagnóstico
├── requirements.txt               # Dependencias
├── README.md                      # Este archivo
├── USAGE.md                       # Guía de uso detallada
└── env/                           # Virtualenv
```

## 📚 Referencias

- [3GPP TS 38.211](https://www.3gpp.org/DynaReport/38211.htm) - Physical channels and modulation
- [3GPP TS 38.213](https://www.3gpp.org/DynaReport/38213.htm) - Physical layer procedures
- [UHD Manual](https://files.ettus.com/manual/)
- [MATLAB 5G Toolbox](https://www.mathworks.com/products/5g.html)

## 📝 Licencia

[Especificar licencia]

## 👤 Autor

[Tu nombre/organización]

## 🤝 Contribuciones

Reportar issues o sugerencias en [repositorio].
