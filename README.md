# Monitoreo Continuo 5G NR con USRP B210

Sistema de monitoreo y análisis de señales 5G NR usando USRP B210 en Python.

## Requisitos

### Hardware
- USRP B210 (Ettus Research)
- Antena compatible con banda 5G (3.3-3.8 GHz para n78)
- Cable USB 3.0

### Software
- Ubuntu 20.04+ (o distribución compatible)
- Python 3.8+
- UHD (USRP Hardware Driver)

## Instalación

### 1. Instalar UHD del sistema

```bash
sudo apt update
sudo apt install -y python3-uhd uhd-host libuhd-dev
sudo uhd_images_downloader
```

### 2. Configurar entorno Python

```bash
# Crear entorno virtual
python3 -m venv env
source env/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Enlazar UHD al entorno virtual
ln -s /usr/lib/python3/dist-packages/uhd env/lib/python3.*/site-packages/uhd
```

### 3. Verificar instalación

```bash
# Listar dispositivos USRP
python monitoreo_continuo.py --list-devices

# Probar modo simulación
python monitoreo_continuo.py --simulate --no-gui --monitor-time 0.1
```

## Uso

### Listar dispositivos disponibles

```bash
python monitoreo_continuo.py --list-devices
```

Salida ejemplo:
```
=== DISPOSITIVOS USRP DISPONIBLES ===

[0] Dispositivo encontrado:
    type: b200
    serial: 12345678
    name: MyB210
========================================
```

### Uso básico con hardware

```bash
# Usar primer dispositivo (si solo hay uno conectado)
python monitoreo_continuo.py

# Especificar dispositivo por índice
python monitoreo_continuo.py --device-index 0

# Especificar por número de serie
python monitoreo_continuo.py --device-serial 12345678
```

### Configuración RF

```bash
# Cambiar frecuencia usando GSCN (Global Synchronization Channel Number)
python monitoreo_continuo.py --gscn 7929  # 3619.2 MHz (banda n78)

# Ajustar ganancia del receptor
python monitoreo_continuo.py --gain 40  # 40 dB

# Cambiar tasa de muestreo
python monitoreo_continuo.py --sample-rate 23.04e6  # 23.04 MHz

# Configurar subcarrier spacing
python monitoreo_continuo.py --scs 30  # 30 kHz (opciones: 15, 30, 60, 120, 240)
```

### Parámetros de monitoreo

```bash
# Tiempo total de monitoreo
python monitoreo_continuo.py --monitor-time 1.0  # 1 segundo

# Intervalo entre capturas
python monitoreo_continuo.py --interval 0.1  # 100 ms

# Frames por captura
python monitoreo_continuo.py --frames 2  # 2 frames (20 ms)

# Sin visualización gráfica
python monitoreo_continuo.py --no-gui
```

### Modo simulación (sin hardware)

```bash
# Generar datos sintéticos para pruebas
python monitoreo_continuo.py --simulate

# Simulación sin GUI
python monitoreo_continuo.py --simulate --no-gui
```

### Ejemplos completos

```bash
# Monitoreo en banda n78 con ganancia media
python monitoreo_continuo.py --gscn 7929 --gain 45 --monitor-time 2.0

# Múltiples dispositivos - usar el segundo
python monitoreo_continuo.py --device-index 1 --gain 50

# Captura extendida sin visualización
python monitoreo_continuo.py --monitor-time 10 --interval 0.5 --no-gui

# Prueba rápida en modo simulación
python monitoreo_continuo.py --simulate --monitor-time 0.2 --interval 0.1 --no-gui
```

## Argumentos de línea de comandos

### Selección de dispositivo
- `--list-devices`: Lista dispositivos USRP disponibles y sale
- `--device-index N`: Índice del dispositivo a usar (0, 1, 2, ...)
- `--device-serial SERIAL`: Número de serie del dispositivo
- `--device-args ARGS`: Argumentos adicionales del dispositivo

### Configuración RF
- `--gscn GSCN`: Global Synchronization Channel Number (default: 7929)
- `--sample-rate Hz`: Tasa de muestreo en Hz (default: 19.5e6)
- `--gain dB`: Ganancia del receptor en dB (default: 50)
- `--scs {15,30,60,120,240}`: Subcarrier spacing en kHz (default: 30)

### Parámetros de monitoreo
- `--monitor-time s`: Tiempo total de monitoreo en segundos (default: 0.57)
- `--interval s`: Intervalo entre capturas en segundos (default: 0.057)
- `--frames N`: Número de frames por captura (default: 1)

### Otros
- `--simulate`: Modo simulación (sin hardware)
- `--no-gui`: Desactivar visualización gráfica
- `-h, --help`: Mostrar ayuda

## Frecuencias 5G NR (FR1)

### Banda n78 (3.3-3.8 GHz)
| GSCN | Frecuencia (MHz) | Descripción |
|------|------------------|-------------|
| 7499 | 3000.00 | Inicio banda |
| 7929 | 3619.20 | Común en Europa |
| 8065 | 3815.04 | Común en Asia |
| 8255 | 4088.64 | Fin banda n78 |

Cálculo: `Freq(MHz) = 3000 + (GSCN - 7499) × 1.44`

## Estructura del código

- **USRPB210Receiver**: Clase para control del hardware USRP
  - Configuración de frecuencia, ganancia, tasa de muestreo
  - Captura de muestras IQ
  
- **NR5GProcessor**: Clase para procesamiento de señales 5G NR
  - Corrección de frecuencia
  - Demodulación OFDM
  - Detección de SSB (Synchronization Signal Block)
  - Identificación de Cell ID
  
- **Funciones principales**:
  - `list_usrp_devices()`: Lista dispositivos disponibles
  - `select_usrp_device()`: Selecciona dispositivo específico
  - `capture_waveforms()`: Captura múltiples waveforms
  - `demodulate_all()`: Demodula todos los waveforms
  - `visualize_resource_grids()`: Visualiza resultados

## Solución de problemas

### No se encuentra el dispositivo
```bash
# Verificar conexión USB
lsusb | grep Ettus

# Listar con verbose
python monitoreo_continuo.py --list-devices

# Verificar permisos
sudo usermod -a -G usrp $USER  # Reiniciar sesión después
```

### Error de importación de UHD
```bash
# Verificar instalación
python -c "import uhd; print('UHD OK')"

# Reinstalar si es necesario
sudo apt install --reinstall python3-uhd

# Verificar enlace simbólico
ls -l env/lib/python3.*/site-packages/uhd
```

### Error de compatibilidad NumPy
```bash
# UHD requiere NumPy 1.x
pip install "numpy<2"
```

### Imagen de firmware no encontrada
```bash
# Descargar imágenes del firmware
sudo uhd_images_downloader

# Verificar variable de entorno
echo $UHD_IMAGES_DIR
export UHD_IMAGES_DIR=/usr/share/uhd/images
```

## Salida del programa

El programa genera las siguientes métricas:

- **Potencia**: Potencia promedio de la señal en dB
- **SNR**: Relación señal-ruido estimada en dB
- **Cell ID**: Physical Cell ID detectado (0-1007)
- **Resource Grid**: Visualización del grid de recursos OFDM

### Ejemplo de salida:
```
=== RESUMEN DE TIEMPOS ===
Captura promedio: 0.023s (total: 0.231s)
Demodulación promedio: 1.245s (total: 12.450s)
Visualización: 0.105s

=== MÉTRICAS DE SEÑAL ===
Potencia promedio: -45.3 dB
SNR promedio: 12.7 dB
Cell IDs detectados: {156, 157}
```

## Referencias

- [3GPP TS 38.211](https://www.3gpp.org/DynaReport/38211.htm) - Physical channels and modulation
- [Ettus Research UHD Documentation](https://files.ettus.com/manual/)
- [5G NR Specifications](https://www.3gpp.org/specifications-technologies/releases/release-15)

## Licencia

Este proyecto es parte de trabajos de investigación y monitoreo de redes 5G.

## Autor

Desarrollado para análisis y monitoreo de señales 5G NR en tiempo real.
