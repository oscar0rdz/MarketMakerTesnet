# 🚀 Guía Rápida - Correcciones Implementadas

## ✅ Problemas Resueltos

1. **Error de cantidad mínima** (BTC/SOL)
2. **Error de autenticación intermitente**
3. **Logging mejorado** para debugging

---

## 🔧 Antes de Ejecutar

### 1. Verificar Configuración
```bash
python3 check_sizes.py
```

Este script verifica:
- ✅ API Keys cargadas correctamente
- ✅ Conexión a Binance Testnet
- ✅ Tamaños configurados vs mínimos de Binance
- ✅ Precios actuales y cantidades calculadas

**Ejemplo de output:**
```
================================================================================
🔍 DIAGNÓSTICO DE CONFIGURACIÓN - Trading Bot
================================================================================

📋 1. Verificando API Keys...
   ✅ API Key: FdRUWIvnIF...8uYd4el8
   ✅ API Secret: zhdvMGMy26...fEW8zL9R6

📡 2. Conectando a Binance Testnet...
   ✅ Testnet/Sandbox habilitado
   ✅ Mercados cargados correctamente

📊 3. Verificando Tamaños Configurados vs Mínimos de Binance...
--------------------------------------------------------------------------------

   ✅ BTCUSDT:
      Config: $110.00 USDT → 0.00112100 qty
      Mínimo: 0.001 qty
      Precio: $98143.20
      Step: 0.001

   ✅ SOLUSDT:
      Config: $250.00 USDT → 1.04000000 qty
      Mínimo: 1 qty
      Precio: $240.38
      Step: 1

================================================================================
✅ TODOS LOS CHECKS PASARON - Bot listo para ejecutar
================================================================================
```

---

## 🚀 Ejecutar el Bot

### Modo Normal:
```bash
python3 main.py
```

### Modo Debug (recomendado primera vez):
```bash
LOG_LEVEL=DEBUG python3 main.py
```

### Output Esperado:
```
================================================================================
🚀 Starting real Binance scalper
Pairs: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
================================================================================
✅ API credentials loaded successfully
🧪 Sandbox/Testnet mode enabled.

🟢 APERTURA BTCUSDT | LONG | Precio: 98143.200000 | TP: 1.50% | SL: 0.75% | ATR: 0.0245 | Régimen: TRENDING
[BTCUSDT] Placing BUY order: size=0.001121 (normalized from 0.001121)
```

---

## 📁 Archivos Importantes

### Documentación:
- **`FINAL_SUMMARY.md`** - Resumen completo de cambios
- **`FIX_MINIMUM_SIZES.md`** - Detalles técnicos
- **`LOGGING_README.md`** - Sistema de logging (anterior)
- **`QUICK_START.md`** - Este archivo

### Scripts:
- **`check_sizes.py`** - Diagnóstico pre-ejecución
- **`main.py`** - Bot principal (corregido)
- **`analyze_trades_example.py`** - Análisis de trades

### Configuración:
- **`.env`** - Variables de entorno
- **`psar_scalper/src/config.py`** - Config de pares (corregido)

---

## 🔍 Si Hay Errores

### Error: "amount must be greater than minimum"

**Solución:**
1. Ejecuta `python3 check_sizes.py`
2. Aumenta el `base_size_usdt` en `.env`:
   ```env
   BTCUSDT_BASE_SIZE_USDT=120.0
   SOLUSDT_BASE_SIZE_USDT=300.0
   ```

### Error: "requires apiKey credential"

**Solución:**
1. Verifica que `.env` esté en el directorio raíz
2. Verifica contenido:
   ```bash
   cat .env | grep BINANCE_API
   ```
3. Las keys deben ser de **testnet** (https://testnet.binancefuture.com)

### Ver logs detallados:

```bash
LOG_LEVEL=DEBUG python3 main.py 2>&1 | tee bot.log
```

Busca líneas específicas:
```bash
# Ver cálculo de cantidades
grep "notional_to_size" bot.log

# Ver órdenes ejecutadas
grep "Placing" bot.log

# Ver errores
grep "ERROR" bot.log
```

---

## 📊 Analizar Resultados

Después de que el bot haya ejecutado trades:

```bash
python3 analyze_trades_example.py
```

Verás:
- Win rate
- PnL total y promedio
- Análisis por par
- Análisis por régimen
- Top mejores/peores trades

---

## ⚙️ Ajustar Tamaños

### Por variable de entorno (`.env`):
```env
BTCUSDT_BASE_SIZE_USDT=120.0
ETHUSDT_BASE_SIZE_USDT=25.0
SOLUSDT_BASE_SIZE_USDT=300.0
BNBUSDT_BASE_SIZE_USDT=20.0
XRPUSDT_BASE_SIZE_USDT=15.0
```

### Por código (`psar_scalper/src/config.py`):
```python
"BTCUSDT": PairConfig(
    symbol="BTCUSDT",
    base_size_usdt=120.0,  # Cambiar aquí
    scale_size_usdt=60.0,
    # ...
)
```

**Tip:** Usa `.env` para cambios temporales, `config.py` para permanentes.

---

## ✅ Checklist

Antes de ejecutar en producción:

- [ ] Ejecuté `check_sizes.py` sin errores
- [ ] Probé en testnet primero
- [ ] Revisé logs con `LOG_LEVEL=DEBUG`
- [ ] Verifiqué que los trades se ejecutan correctamente
- [ ] Analicé resultados con `analyze_trades_example.py`
- [ ] Ajusté tamaños si fue necesario

---

## 🆘 Soporte

Si sigues teniendo problemas:

1. **Ejecuta diagnóstico:**
   ```bash
   python3 check_sizes.py > diagnostic.txt 2>&1
   ```

2. **Ejecuta bot con debug:**
   ```bash
   LOG_LEVEL=DEBUG python3 main.py > bot_debug.log 2>&1
   ```

3. **Revisa ambos archivos** para identificar el problema

---

**🎉 Todo listo! Comienza con `python3 check_sizes.py` y luego `python3 main.py`**
