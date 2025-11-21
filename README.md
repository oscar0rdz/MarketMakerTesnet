# PSAR Scalping Bot para Binance Futures

<div align="center">
  <h3>Demo del PSAR Scalping Bot en Binance Futures</h3>
  <a href="https://www.youtube.com/watch?v=Ke04Dqzy6mM" target="_blank">
    
         alt="PSAR Scalping Bot - Demo en YouTube"
         style="max-width: 100%; border-radius: 12px;" />
  </a>
  <p>
    Haz clic en la imagen para ver el video de demostración en YouTube.
  </p>
</div>
Bot de **scalping multi–par** para Binance Futures (testnet o producción), construido en Python, que combina:

- **PSAR en 30m** como cerebro estructural de tendencia.
- **Market Bias + Twin Range Filter (TRF) en 5m** como filtros de contexto y estructura.
- **Ejecución en 1m** con gestión micro de salidas.
- **Sesgo macro con EMA 200 en 4H** para decidir si favorecer LONGs o SHORTs.
- **Sistema de score y planner ALWAYS_IN_MARKET**, con pesos dinámicos por ADX y tamaño de posición ligado a la calidad de la señal.

El proyecto sirve  como **laboratorio de trading cuantitativo** , donde cada decisión (indicadores, timeframes, riesgo, fees) está conectada con código real y medible.

---

## 1. Visión general del proyecto

El bot busca mantener una **exposición constante pero controlada** a varios pares de futuros:


- Usa un **notional base** por trade (por ejemplo, `70 USDT`) con apalancamiento configurable (ej. `10x`).
- Selecciona solo los pares que muestran **mejores condiciones de mercado** según un sistema de **score macro/micro**.
- Ajusta **SL/TP y tamaño** en función de la **volatilidad (ATR)** y del **régimen del mercado** (tendencia fuerte vs rango).

El objetivo no es prometer una estrategia “mágica”, sino mostrar un sistema:

- Modular, legible y extensible.
- Con un **modelo de riesgo explícito**.
- Diseñado para ser probado críticamente mediante **logs y backtests**.

---

## 2. Pregunta de investigación y enfoque

> ¿Puede una estrategia basada en PSAR 30m, filtros de Market Bias + Twin Range Filter en 5m y un sistema de scoring adaptativo (ADX, ATR, EMA200 4H) compensar comisiones y volatilidad en Binance Futures usando posiciones pequeñas y reglas de riesgo sistemáticas?

Para responderla, el bot integra:

- **PSAR 30m** como referencia estructural de tendencia.
- **MB + TRF 5m** para discriminar entre entornos tendenciales y de rango.
- **Sesgo macro EMA200 4H** para decidir cuándo favorecer LONGs o SHORTs.
- **Score dinámico** que decide:
  - si vale la pena operar,
  - cuánto tamaño (notional) asumir,
  - en qué símbolos concentrar el riesgo.
.

---

## 3. Arquitectura multi–timeframe

La estrategia se apoya en 4 escalas:

1. **4H (macro–bias)**  
   - **EMA 200**: si el precio está por encima → sesgo alcista; si está por debajo → sesgo bajista.  
   - Este sesgo se usa para **favorecer LONGs en mercados alcistas** y **SHORTs en mercados bajistas**.

2. **30m (macro de trading)**  
   - **PSAR 30m**: define la dirección base de tendencia.
   - **EMAs 30m y ADX 30m**: miden dirección y **fuerza** de la tendencia.
   - **ATR% 30m**: clasifica el régimen de volatilidad (baja, normal, alta).

3. **5m (micro–estructura)**  
   - **Market Bias 5m**: sesgo direccional suavizado (Heikin Ashi + EMAs).
   - **Twin Range Filter 5m**:
     - clasifica si el mercado está en **tendencia** o **rango**,
     - usa una EMA central + bandas basadas en ATR.

4. **1m (ejecución y gestión)**  
   - PSAR y EMAs rápidas en 1m.
   - Confirmación fina de entradas.
   - Gestión de salidas tempranas (deterioro, time–stop, trailing).

**Idea clave:**  
Las decisiones importantes (lado de la operación, tamaño, prioridad) no se toman en 1m aislado: 1m solo ejecuta lo que ya fue filtrado por 4H, 30m y 5m.

---

## 4. Núcleo de la lógica: score y priority 

### 4.1 Score vs Priority

El sistema distingue dos cosas:

- **Score (puntuación pura)**  
  - Mide la **calidad objetiva** de la señal (edge estimado).  
  - Se construye a partir de:
    - alineación con tendencia macro (PSAR/EMAs 30m),
    - estructura micro (MB + TRF 5m),
    - salud de la volatilidad (ATR% 30m),
    - fuerza de la señal primaria.  
  - No tiene sesgo favorito a LONG o SHORT.

- **Priority (prioridad)**  
  - Es el **score ajustado** por sesgos operativos:
    - sesgo EMA200 4H (pro–LONG en bull, pro–SHORT en bear),
    - cooldown después de un trade perdedor,
    - límites por clúster (LARGE_CAP, ALT_CLUSTER, etc.).  
  - Se usa para elegir **qué símbolos se operan primero** cuando hay varios candidatos válidos.

El **engine** filtra primero por score (calidad mínima) y luego ordena por priority (preferencias estratégicas).

---

### 4.2 Macro vs Micro con pesos dinámicos (ADX)

En vez de pesos fijos (ej. 15% macro, 70% micro), el bot utiliza **ponderación dinámica basada en ADX 30m**:

- **Tendencia fuerte (ADX alto)**  
  - Se aumenta el peso del componente **macro**:
    - seguir la dirección del tren (tendencia) importa más que el ruido micro.
- **Rango / lateral (ADX bajo)**  
  - Se aumenta el peso del componente **micro**:
    - importa más el timing fino (reversiones, rebotes, microestructura).

Esto permite que la misma lógica se adapte a **días tendenciales** y **días laterales** sin cambiar manualmente parámetros.

---

### 4.3 Sesgo adaptativo con EMA200 4H (bias pro–trend)

El sesgo pro–LONG ya no es fijo:

- **Si el precio está por encima de EMA200 4H**  
  - Se favorecen **LONGs** (priority se bonifica).
  - Los SHORTs se penalizan en prioridad (pero no se prohíben).

- **Si el precio está por debajo de EMA200 4H**  
  - Se favorecen **SHORTs**.
  - Los LONGs se penalizan.

Así, el sistema deja de luchar contra la corriente en mercados bajistas y puede capturar mejor las caídas cuando el contexto lo respalda.

---

### 4.4 ATR: de “veto” a escalado de tamaño

Antes, volatilidades extremas (ATR% muy baja o muy alta) podían **matar la señal** (`score = 0`).  
Ahora, la lógica es más fina:

- Se sigue calculando una **“salud” de ATR% 30m** (punto dulce ≈ volatilidad normal).
- Pero en lugar de anular la señal:
  - el score se mantiene,
  - y lo que se ajusta es el **tamaño de la posición** (`risk_mult`).

Ejemplo simplificado:

- ATR normal → `risk_mult ≈ 1.0` (usa el notional base).
- ATR muy alta → `risk_mult` se reduce (ej. 0.3–0.5).
- ATR muy baja → el bot puede reducir también tamaño o filtrar por RRR mínimo útil.

**Consecuencia:**  
El bot puede aprovechar eventos de alta volatilidad con **posiciones pequeñas**, en lugar de quedarse siempre fuera.

---

### 4.5 Score = confianza = tamaño (RRR constante)

El **RRR (riesgo/beneficio)** se mantiene prácticamente **constante** (ej. `1.45`) y el score se usa sobre todo para decidir **cuánto capital arriesgar**:

- Score bajo (señal mediocre pero aceptable)  
  - `risk_mult` cercano a 0.3–0.5  
  - Tamaño pequeño, RRR ~ 1.45.

- Score alto (señal muy clara)  
  - `risk_mult` cercano a 0.8–1.0 (o el máximo configurado)  
  - Tamaño grande, mismo RRR ~ 1.45.

La lógica implícita es:

> Una señal de alta calidad es algo que probablemente va a salir bien.  
> En lugar de perseguir targets lejanos, se apuesta **más tamaño a targets razonables**.

---



- **Objetivos principales:**
  - Mantener al menos `TARGET_MIN_OPEN_SYMBOLS` posiciones (ej. 4).
  - No exceder `MAX_OPEN_SYMBOLS` (ej. 5).
  - Respetar límites por símbolo y clúster.

Comportamiento:

1. **Déficit (open_count < target_min)**  
   - Calcula un **umbral dinámico** a partir del mejor score (`max_abs_score`).
   - Puede pasar de modo **estricto** a modo **soft–fill** si pasan muchas iteraciones sin llenar slots.
   - Abre nuevas posiciones hasta recuperar el mínimo.

2. **Estable (open_count ≥ target_min)**  
   - Se mantiene en modo **estricto**.
   - Solo abre nuevas posiciones si se libera un slot (cierre por SL/TP/time–stop).
   - Opcionalmente, puede usar un **slot “premium”** para una señal excepcional.

En la práctica, el planner mantiene una cartera viva, pero exige **calidad mínima** y gestiona el ritmo de nuevas entradas para no saturar.

---

## 5. Indicadores y lógica por timeframe

### 5.1 4H: EMA 200 (bias macro)

- Determina si el entorno general es:
  - **Bull** (precio > EMA200),
  - **Bear** (precio < EMA200).
- Este bias impacta directamente en la **priority** y en algunos límites de riesgo.

### 5.2 30m: PSAR, EMAs, ATR, ADX

- **PSAR 30m**:
  - Por debajo del precio → contexto alcista.
  - Por encima del precio → contexto bajista.
  - Flip → evento relevante para cerrar/revertir.

- **EMAs 30m**:
  - Cruces y pendientes apoyan el régimen de tendencia.

- **ATR% 30m**:
  - Se transforma en `% del precio` para clasificar volatilidad y escalar tamaño.

- **ADX 30m**:
  - ADX alto → tendencia fuerte (más peso al macro).
  - ADX bajo → rango (más peso al micro).

### 5.3 5m: Market Bias y Twin Range Filter

- **Market Bias 5m**:
  - Basado en Heikin Ashi + EMAs.
  - Entrega una señal suavizada (+1, 0, -1).

- **Twin Range Filter 5m**:
  - EMA central + bandas ATR.
  - Clasifica:
    - entorno tendencial (precio fuera de bandas y pendiente clara),
    - entorno de rango/chop (precio dentro, pendiente plana).

### 5.4 1m: ejecución táctica

- Confirma la entrada con PSAR + EMAs rápidas.
- Gestiona:
  - **salidas tempranas** por deterioro,
  - **time stops** según régimen de volatilidad,
  - **trailing stops** cuando la operación avanza a favor.

---

## 6. Arquitectura técnica

### 6.1 Stack tecnológico

- **Python 3.10+**
- `pandas`, `numpy` para series temporales.
- `ccxt` para conexión con Binance Futures (incluida testnet).
- `python-dotenv` para manejo de `.env`.
- `logging` estándar + CSV (`logs/trades.csv`) para análisis.
- Tests unitarios para partes críticas (indicadores, sizing, etc.).

### 6.2 Estructura del repositorio (resumen)

```bash
MarketMakerTesnet/
├── main.py                      # Punto de entrada
├── logger_setup.py              # Configuración de logs
├── requirements.txt
├── .env                         # Config local (no se versiona)
│
├── psar_scalper/
│   └── src/
│       ├── config.py            # EngineSettings, ScoreSettings, planner
│       ├── data.py              # DataFeed (velas 1m, 5m, 30m, 4H)
        ├── engine.py            # Bucle principal, planner ALWAYS_IN_MARKET
        ├── indicators.py        # PSAR, EMAs, ATR, ADX, etc.
        ├── scoring.py           # SymbolSignals, EntryCandidate, score → riesgo
        ├── risk.py              # build_sl_tp, RRR constante
        ├── state.py             # Dataclasses, posiciones, PlannerState
        ├── mb_trf_indicators.py # Market Bias y Twin Range Filter (5m)
        └── trend.py             # Régimen macro 30m
│
├── logs/
│   └── trades.csv               # Historial de operaciones
│
└── tests/
    └── test_position_size.py    # Ejemplo de tests de sizing
```

*(Los nombres pueden variar ligeramente según la rama, pero la idea modular se mantiene.)*

---

## 7. Configuración (`.env`)

Ejemplo de archivo `.env`:

```bash
# Credenciales de Binance Futures
BINANCE_API_KEY=TU_API_KEY
BINANCE_API_SECRET=TU_API_SECRET
BINANCE_FUTURES_TESTNET=true        # true = testnet, false = producción

# Parámetros de trading
SYMBOLS=XRPUSDT,BNBUSDT,LTCUSDT,LINKUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,XLMUSDT
BASE_NOTIONAL_USDT=70.0
LEVERAGE=10

# Gestión de portafolio
TARGET_MIN_OPEN_SYMBOLS=4
MAX_OPEN_SYMBOLS=5

# Riesgo y fees
FEE_ROUNDTRIP_PCT=0.0010            # ida y vuelta
EDGE_MIN_PCT=0.0020                 # edge mínimo neto
SL_MIN_PCT=0.0040                   # stop mínimo
BASE_RRR=1.45                       # R:R base (constante)

# Score / planner
SCORE_THRESHOLD=0.25
SOFT_SCORE_THRESHOLD=0.18
SOFT_FILL_AFTER_EMPTY_ITERS=5

# Logging
LOG_LEVEL=INFO
```

Ajusta los valores según tu apetito de riesgo y el capital que quieras simular.

---

## 8. Ejecución

### 8.1 Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/MarketMakerTesnet.git
cd MarketMakerTesnet
```

### 8.2 Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scriptsctivate
pip install -r requirements.txt
```

### 8.3 Ejecutar el bot (testnet)

```bash
python main.py
```

El bot:

- Sincroniza tiempo y posiciones con Binance.
- Descarga velas 4H, 30m, 5m, 1m.
- Calcula indicadores y scores.
- Abre/cierra posiciones según la lógica `ALWAYS_IN_MARKET` y la configuración de riesgo.

---

## 9. Logging y análisis de resultados

El archivo `logs/trades.csv` registra para cada trade:

- Timestamp
- Símbolo
- Dirección (LONG/SHORT)
- Precio de entrada y salida
- Notional y tamaño
- PnL absoluto y porcentual
- Razón de cierre (TP, SL, deterioro, time–stop, etc.)

Esto permite:

- Reconstruir la curva de equity.
- Calcular winrate, profit factor, drawdown, etc.
- Analizar por símbolo, por horario o por régimen de volatilidad.

La consola muestra, además, detalles como:

- Score, priority y `risk_mult` al abrir cada posición.
- RRR aplicado y distancia SL/TP (en % y en ATR).
- Estado del planner: `open_count`, `target_min`, `max_open`.

---

## 10. Limitaciones y líneas de trabajo futuras

Limitaciones actuales:

- Parámetros escogidos de forma razonable, pero no exhaustivamente optimizados.
- Correlación entre pares manejada de forma simple (por clúster), aún mejorable.
- Sin filtros de volumen explícitos.
- No incluye, por defecto, un módulo de backtesting integrado (se apoya en `logs/trades.csv` para análisis posterior).

Posibles mejoras:

- Integrar backtesting/replay sobre `trades.csv` y datos históricos.
- Añadir filtros de **correlación dinámica** entre símbolos.
- Incluir filtros de **volumen mínimo** y horario.
- Implementar **salidas parciales** (take parcial + runner).
- Integrar **notificaciones** (Telegram/Discord) con resúmenes de trades.

---

---

## 12. Glosario

- **PSAR**: Parabolic SAR, indicador tipo stop-and-reverse.
- **EMA**: Exponential Moving Average.
- **ATR**: Average True Range, mide volatilidad.
- **ADX**: Average Directional Index, mide fuerza de tendencia.
- **Market Bias (MB)**: Señal de sesgo direccional suavizado en 5m.
- **Twin Range Filter (TRF)**: Filtro que clasifica tendencia vs rango usando una línea media y bandas.
- **RRR (R:R)**: Relación riesgo/beneficio.
- **SL**: Stop Loss.
- **TP**: Take Profit.
- **Notional**: Valor nominal de la posición en USDT.
- **Régimen**: Estado de volatilidad/tendencia del mercado.
- **ALWAYS_IN_MARKET**: Enfoque donde el bot busca mantener un mínimo de posiciones abiertas de forma casi continua.

---

## 13. Estado del proyecto y propósito como portafolio

Este bot está pensado como:

- **Herramienta de experimentación** en scalping cuantitativo con derivados.
- **Ejemplo de arquitectura modular** en Python para trading en tiempo real.
- **Pieza de portafolio**, donde se pueda revisar:
  - cómo se diseña un sistema de score macro/micro,
  - cómo se enlaza con riesgo, fees y límites operativos,
  - cómo se documenta y registra todo para evaluación crítica.

No constituye recomendación financiera.  
La responsabilidad final de probar, entender y ajustar la estrategia recae en quien la ejecute, idealmente empezando siempre por **testnet** y análisis detallado de `logs/trades.csv`.
