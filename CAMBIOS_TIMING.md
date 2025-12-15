# Mejoras en Timing Estimation - Monitoreo 5G NR

## Resumen de Cambios

Se ha reimplementado completamente la detección y posicionamiento del SSB para que aparezca **siempre a la izquierda** del resource grid, siguiendo exactamente el proceso de MATLAB.

## Problema Original

El SSB no aparecía consistentemente en la misma posición del resource grid. El código no estaba **recortando** correctamente la señal después de detectar el timing offset.

## Cambios Implementados

### 1. Nueva función `_timing_estimate()` mejorada

Implementación completa equivalente a `nrTimingEstimate` de MATLAB:

**Proceso (siguiendo ejemplo MATLAB):**
1. Crea un **reference grid de 4 símbolos** con PSS:
   ```python
   # Equivalente a: txGrid(pssInd) = pssSym
   ref_grid = zeros([240, 4])  # 20 RBs x 4 símbolos SSB
   ref_grid[center-63:center+64, 1] = nrPSS(NID2)  # PSS en símbolo 1
   ```

2. **OFDM Modulate** el reference grid a dominio temporal:
   ```python
   # Equivalente a: txWaveform = nrOFDMModulate(carrier, txGrid)
   for sym in range(4):
       time_sym = IFFT(ref_grid[:, sym]) * fft_size
       ofdm_sym = [CP, time_sym]
       ref_time[sym*symbol_length:(sym+1)*symbol_length] = ofdm_sym
   ```

3. **Correlación cruzada** con la señal recibida:
   ```python
   # Equivalente a: offset = nrTimingEstimate(rxWaveform, ...)
   correlation = np.correlate(waveform, ref_time, mode='valid')
   timing_offset = argmax(abs(correlation))
   ```

4. El **pico indica el inicio exacto del SSB** en la señal

### 2. Proceso de Detección SSB Corregido

La función `find_ssb()` ahora implementa correctamente el **recorte de señal**:

```
PASO 1: Frequency Correction (hSSBurstFrequencyCorrect)
    ↓
PASO 2: Timing Estimation (nrTimingEstimate)
    → Devuelve el INICIO EXACTO del SSB en muestras
    ↓
PASO 3: RECORTAR la señal desde timing_offset
    → corrected_waveform = corrected_waveform[timing_offset:]
    → ¡CLAVE! SSB ahora empieza en muestra 0
    ↓
PASO 4: OFDM Demodulation (desde muestra 0)
    → SSB está en los primeros 4 símbolos [0:4]
    ↓
PASO 5: Extraer símbolos SSB (símbolos 0-3)
    → grid_ssb = grid[:, 0:4]
    ↓
PASO 6: SSS Correlation (detectar NID1)
    ↓
PASO 7: PBCH DM-RS Analysis (determinar SSB index)
    ↓
PASO 8: Demodular grid completo para visualización
    → La señal YA está recortada
    ↓
PASO 9: SSB aparece en símbolos [0:4]
    → A la IZQUIERDA del resource grid
```

### 3. Corrección CLAVE: Recorte de Señal

**Antes:**
- Timing offset se calculaba pero NO se aplicaba correctamente
- SSB aparecía en posiciones variables del grid
- Rectángulo SSB mal posicionado

**Ahora:**
- **PASO CRÍTICO**: `corrected_waveform = corrected_waveform[timing_offset:]`
- Esto hace que el SSB empiece en la **muestra 0** de la señal recortada
- SSB siempre aparece en **símbolos 0-3** (izquierda del grid)
- Rectángulo SSB: `[0.5, ssb_freq_origin-0.5, 4, 240]`

**Analogía con MATLAB:**
```matlab
% MATLAB hace esto:
offset = nrTimingEstimate(rxWaveform, ...);
rxWaveform = rxWaveform(1+offset:end);  % ← RECORTE
rxGrid = nrOFDMDemodulate(rxWaveform, ...);
% SSB está en primeros símbolos del grid
```

### 4. Correlación SSS Mejorada

Cambio en la fórmula de correlación para coincidir con MATLAB:

**MATLAB:**
```matlab
sssEst(NID1+1) = sum(abs(mean(sssRx .* conj(sssRef),1)).^2);
```

**Python equivalente:**
```python
corr = np.abs(np.mean(sss_rx * np.conj(sss_ref)))**2
```

Esto mejora la precisión en la detección del NID1.

## Resultados

### Modo Simulación
```
Timing offset: 9316 muestras (inicio del SSB)
Señal recortada: 380684 muestras
SSB ahora empieza en muestra 0
SSB en grid: símbolos [0:4], subportadoras [61:301]
NID1: 44, Cell ID: 133, SSS corr: 2.74e+01, Detected: True
```

### Captura Real (USRP B210)
```
Timing offset: 167157 muestras (inicio del SSB)
Señal recortada: 222843 muestras
SSB ahora empieza en muestra 0
SSB en grid: símbolos [0:4], subportadoras [61:301]  ← A la IZQUIERDA
NID1: 13, Cell ID: 41, SSS corr: 2.26e-01, Detected: True
Potencia: -38.3 dB, SNR: 8.0 dB
```

### Visualización

El SSB ahora aparece **consistentemente a la izquierda** de cada resource grid:
- **Símbolos 0-3**: SSB (4 símbolos)
- **Símbolos 4-13**: Resto del slot
- **Rectángulo rojo**: Posicionado correctamente sobre el SSB

## Compatibilidad

- ✅ Totalmente compatible con configuración YAML existente
- ✅ Mantiene todos los argumentos CLI
- ✅ Sin cambios en API pública
- ✅ Modo simulación y hardware funcionan correctamente

## Referencias

- **MATLAB nrTimingEstimate:** https://es.mathworks.com/help/5g/ref/nrtimingestimate.html
- **3GPP TS 38.211:** Physical channels and modulation
- **Implementación original:** `MonitoreoContinuoFunciones.m`

## Ejemplo de Uso

```bash
# Con captura real y visualización
env/bin/python monitoreo_continuo.py --device-serial 3273A5C --verbose

# Modo simulación para pruebas
env/bin/python monitoreo_continuo.py --simulate --verbose --no-gui

# Con configuración personalizada
env/bin/python monitoreo_continuo.py --config config.yaml --n-symbols-display 14
```

---
**Fecha:** 15 de diciembre de 2025
**Versión:** 2.0 (Timing Estimation mejorado)
