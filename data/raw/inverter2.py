"""
inverter2.py
------------
Grafica, para un intervalo fijo de ejemplo (un dia, a 5 min), la
irradiancia estimada a partir de la potencia AC del inversor de la
estacion terrestre, con GHI = P / (eta * N * A), eta = 0.2133,
N = 30 modulos y A = 2.56 m2 por modulo.

Entrada: y.csv (columnas 'date' y 'potencia', en este mismo directorio).
Salida: ninguna en disco; muestra la figura en pantalla.

Como ejecutarlo (con el entorno virtual activo, desde data/raw/):
    python inverter2.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Cargar datos desde y.csv
# ----------------------------------------------------
power = pd.read_csv("y.csv", parse_dates=['date'])

# Usar la columna 'date' como índice
power = power.set_index('date')

# Asegurar zona horaria UTC
if power.index.tz is None:
    power.index = power.index.tz_localize('UTC')
# ----------------------------------------------------
# 2. Definir el intervalo de tiempo a graficar
# ----------------------------------------------------
times = pd.date_range('2023-07-02 00:05', '2023-07-03 00:00', freq='5min', tz='UTC')

start = times.min()
end = times.max()

# Filtrar datos dentro del intervalo
#power = power.loc[start:end]/1000  # Convertir W a kW
#Irradiancia medida (W/m2) = potencia (W) / (0.2133*30*2.56)
power = power.loc[start:end]/(0.2133*30*2.56) # GHI (W/m2) = potencia (W) / (eta*N*A)

# Opcional: energía total (kWh) del día
#energy_kWh = power['potencia'].sum() * (5/60)  # 5 min pasos -> horas fracción
#print("Energía diaria generada (kWh/día):", energy_kWh)

# ----------------------------------------------------
# 3. Graficar la irradiancia estimada
# ----------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(power.index, power['potencia'])  

plt.xlabel('Tiempo (UTC)')
plt.ylabel('Irradiancia Medida (W/m2)')
plt.title('Irradiancia medida vs Tiempo')
plt.grid(True)
plt.tight_layout()
plt.show()
