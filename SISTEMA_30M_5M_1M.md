# Sistema de Trading Multi-Timeframe 30m-5m-1m

## Fecha: 16 de noviembre de 2025

## Resumen Ejecutivo

Este sistema combina tres timeframes (30m, 5m, 1m) con PSAR como ancla de tendencia y Market Bias multinivel para generar un score ponderado que determina la dirección del trading.

---

## 1. Arquitectura del Sistema

### 1.1 Componentes por Timeframe

| Timeframe | Indicador | Propósito | Peso en Score |
|-----------|-----------|-----------|---------------|
| **30m** | PSAR | Tendencia macro y trailing | **3** |
| **5m** | Market Bias (HA + EMA) | Dirección intradía | **2** |
| **1m** | Market Bias (HA corto) | Microestructura direccional | **1** |
| **1m** | Twin Range Filter | Timing y pullbacks | N/A |

### 1.2 Qué Mide Cada Timeframe

#### 30m - Ancla de Tendencia
- **PSAR**: +1 si cierre > SAR, -1 si cierre < SAR
- Define la tendencia macro
- Peso más alto en el score (3x)

#### 5m - Estructura Intradía
- **Market Bias 5m**: Heikin Ashi + EMA suave
  - +1 si HA/EMA alcista
  - -1 si bajista
  - 0 si casi plano
- Confirma o corrige la tendencia 30m
- Peso medio en el score (2x)

#### 1m - Microestructura
- **Market Bias 1m**: HA corto para dirección inmediata
  - +1 / -1 / 0
- **Twin Range Filter 1m**: Timing de entradas
  - Banda verde/roja
  - Entradas en pullbacks a la banda
- Peso menor en el score (1x)

---

## 2. Sistema de Scoring

### 2.1 Cálculo del Score

```python
score = 3 * psar30 + 2 * bias5m + 1 * bias1m
```

**Ejemplo:**
- PSAR 30m = +1 (alcista)
- Bias 5m = +1 (alcista)
- Bias 1m = +1 (alcista)
- **Score = 3(1) + 2(1) + 1(1) = +6** → LONG fuerte

### 2.2 Umbrales de Decisión

| Rango de Score | Interpretación | Acción |
|----------------|----------------|--------|
| **score ≥ +4** | Tendencia LONG fuerte | Abrir/mantener LONG |
| **score ≤ -4** | Tendencia SHORT fuerte | Abrir/mantener SHORT |
| **-3 ≤ score ≤ +3** | Zona gris (indecisión) | Mantener posición actual |

### 2.3 Lógica de Dirección Deseada

```python
if score >= 4:
    desired_dir = +1  # LONG
elif score <= -4:
    desired_dir = -1  # SHORT
else:
    # Zona gris
    if hay_posicion:
        desired_dir = direccion_actual  # NO cambiar
    else:
        desired_dir = psar30  # Fallback a PSAR
```

---

## 3. Reglas de Trading

### 3.1 ALWAYS_IN_MARKET

✅ **Activado**: El bot siempre busca estar en mercado
- Si `state.position is None` y `desired_dir != 0` → Abre posición inmediatamente
- No espera señales perfectas, actúa con consenso mínimo

### 3.2 Mantener Posición

Mientras `posicion.direccion == desired_dir`:
- Recalcula SL/TP según régimen (MICRO/NORMAL/EXPLOSIVE) y ATR
- Permite scale-in solo si:
  - Régimen = NORMAL
  - Twin Range 1m muestra pullback a favor de tendencia

### 3.3 Flip de Posición (Reversal)

Solo hace flip cuando:
```python
if desired_dir != posicion.direccion and abs(score) >= 4:
    # FLIP: cerrar posición actual y abrir en sentido contrario
    cerrar_posicion()
    abrir_posicion(direccion=desired_dir)
```

Si `desired_dir` es opuesto pero `abs(score) < 4`:
- **NO flip**
- Solo aprieta SL (deja que se cierre por stop si el giro se confirma)

### 3.4 Reapertura Rápida

Si cierras por:
- `time_stop` (tiempo máximo en posición)
- `TP` (take profit)

Y en la vela siguiente `desired_dir` sigue igual:
- **ALWAYS_IN_MARKET** hace que entres de nuevo casi inmediatamente
- En tendencias buenas, el bot reaparece tras exits técnicos sin quedarse fuera mucho tiempo

---

## 4. Ventajas del Sistema

### 4.1 Robustez Multi-Timeframe

✅ **PSAR 30m manda**: No se ensucia con ruido de timeframes cortos
✅ **Bias 5m confirma**: Detecta correcciones intradía vs tendencia macro
✅ **Bias 1m afina**: Timing preciso sin sobrerreaccionar

### 4.2 Protección contra Flips Tontos

❌ **No cambia de lado por cualquier flip en 1m**
✅ **Solo flip cuando 30m y 5m están alineados en contra** (score ≥ 4 opuesto)

### 4.3 Maximiza Tiempo en Mercado

✅ **ALWAYS_IN_MARKET**: No se queda fuera esperando señal perfecta
✅ **Reapertura rápida**: Vuelve a entrar tras exits técnicos si tendencia persiste
✅ **Mantiene dirección en zona gris**: No sale por indecisión temporal

---

## 5. Ejemplo de Operación

### Escenario: Tendencia Alcista Fuerte

**Barra 1:**
- PSAR 30m: +1 (alcista)
- Bias 5m: +1 (alcista)
- Bias 1m: +1 (alcista)
- **Score: 3 + 2 + 1 = +6**
- **Acción**: Abrir LONG

**Barra 50:**
- PSAR 30m: +1 (alcista)
- Bias 5m: +1 (alcista)
- Bias 1m: -1 (pullback micro)
- **Score: 3 + 2 - 1 = +4**
- **Acción**: Mantener LONG (score aún +4)

**Barra 100:**
- PSAR 30m: +1 (alcista)
- Bias 5m: 0 (neutral)
- Bias 1m: -1 (bajista)
- **Score: 3 + 0 - 1 = +2**
- **Acción**: Mantener LONG (zona gris, no flip)

**Barra 150:**
- Time stop alcanzado
- **Acción**: Cierra LONG por `time_stop`

**Barra 151:**
- PSAR 30m: +1 (alcista)
- Bias 5m: +1 (alcista)
- Bias 1m: +1 (alcista)
- **Score: +6**
- **Acción**: Reapertura automática LONG

**Barra 200:**
- PSAR 30m: -1 (bajista) ← CAMBIO
- Bias 5m: -1 (bajista)
- Bias 1m: -1 (bajista)
- **Score: -3 - 2 - 1 = -6**
- **Acción**: FLIP a SHORT (score ≤ -4)

---

## 6. Parámetros de Market Bias

### Bias 5m
```python
ema_len=20
osc_len=10
smooth_len=7
```

### Bias 1m
```python
ema_len=14
osc_len=7
smooth_len=5
```

Estos parámetros son ajustables según la volatilidad del par y las pruebas en backtesting.

---

## 7. Integración en el Código

### 7.1 Archivos Modificados

1. **`psar_scalper/src/regime.py`**
   - `RegimeState`: Agregados `psar_dir_30m`, `bias_5m`, `bias_1m`, `intraday_score`, `intraday_dir`
   - `compute_heikin_ashi_bias()`: Función genérica para cualquier timeframe
   - `compute_direction_score_30_5_1()`: Calcula score ponderado
   - `compute_regime_state()`: Ahora genera df_5m y calcula todos los bias

2. **`psar_scalper/src/engine.py`**
   - Score-based direction selection
   - Flip solo con `abs(score) >= 4`
   - Logging detallado con breakdown del score

### 7.2 Flujo de Ejecución

```
1. Calcular PSAR 30m → psar_dir
2. Generar df_5m desde df_1m (resample)
3. Calcular bias_5m (HA + EMA en 5m)
4. Calcular bias_1m (HA + EMA en 1m)
5. Score = 3*psar + 2*bias5m + 1*bias1m
6. Determinar desired_dir según umbrales
7. Aplicar reglas de flip/mantener/abrir
```

---

## 8. Logs de Ejemplo

```
APERTURA TREND XRPUSDT | dir=LONG | score=+6 (PSAR30=+1 B5m=+1 B1m=+1) | 
price=1.382000 | SL=0.50% | TP=1.20% | ATR=0.1234% | regime=UP

FLIP FUERTE: score=-6, pos_dir=1 -> desired_dir=-1

Señal débil opuesta (score=-2), manteniendo posición LONG

CIERRE XRPUSDT | LONG | Entry: 1.382000 -> Exit: 1.395000 | 
PnL: $1.56 (+0.94%) | Razón: time_stop | Barras: 45 | Régimen: NORMAL
```

---

## 9. Próximos Pasos

- [ ] Backtest con datos históricos para validar performance
- [ ] Ajustar parámetros de HA bias según volatilidad por par
- [ ] Implementar `tighten_stop_if_needed()` para zona gris
- [ ] Analizar ratio win/loss con nuevo sistema de scoring
- [ ] Optimizar umbrales de score (¿±4 o ±5?)

---

**Sistema implementado exitosamente el 16 de noviembre de 2025**
