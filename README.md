# SpVAR Homicidios Colombia (Streamlit v7)

## Archivos (raíz del repo GitHub)

- `app.py`
- `requirements.txt`
- `runtime.txt`  (intenta Python 3.12; si la plataforma fuerza 3.14, igual funciona)
- `README.md`

## requirements.txt (importante)

```
numpy>=2.2.0
pandas>=2.2.0
plotly>=5.22.0
streamlit>=1.39.0
```

**No uses** `numpy==1.26.4` en Python 3.14: no hay wheel y se queda en `Building numpy`.

## Streamlit / BudsFresh

1. Push de los 4 archivos a la rama que despliega.
2. Main file: `app.py`
3. Reboot.

Si el log dice `Building numpy`, espera 5–10 min la primera vez en 3.14, o revisa que requirements ya no fije 1.26.4.
