# qc-siteadapt-solar

Control de calidad y *site-adaptation* de datos de recurso solar.

Este repositorio contiene `qcsolar`, un paquete de Python que encadena tres
etapas sobre series de irradiancia: control de calidad bajo el protocolo BSRN,
reconstrucción del dato terrestre a partir del satelital mediante modelos de
regresión, y una nueva aplicación del mismo control de calidad sobre la serie
reconstruida para cuantificar la mejora sobre una base comparable. Incluye
además una aplicación web en Streamlit para inspeccionar y exportar los
diagnósticos de cada etapa.

Es el anexo de código del trabajo de grado *"Análisis del desempeño de la
aplicación de técnicas de control de calidad y site-adaptation en datos de
recurso solar"*, Maestría en Ingeniería Electrónica, Universidad de Nariño
(Pasto, Colombia).

## Qué hace

El procedimiento recibe dos series de la misma localización y resolución
temporal, una satelital y una terrestre, y las procesa en este orden:

1. **Control de calidad.** Marca los registros no confiables con las pruebas
   del protocolo BSRN organizadas en tres niveles: pruebas físicas (límites
   posibles y valores infrecuentes, irradiancia nocturna, sensor caído),
   pruebas expertas (cierre radiativo, *K-tests*, índice de cielo despejado) y
   pruebas de serie temporal (persistencia, tasa de cambio, simetría AM/PM).
   Sobre el resultado aplica la regla de islas y calcula el porcentaje de dato
   utilizable.

2. **Reconstrucción por *site-adaptation*.** Invierte el rol de referencia
   habitual: usa la serie satelital, que no tiene interrupciones, para estimar
   y completar el dato terrestre. Compara nueve modelos de cinco familias
   (referencia sin corregir, regresión lineal simple y multivariable, mapeo
   cuantílico empírico, *Random Forest*, *Extra Trees*, *Gradient Boosting*,
   *XGBoost* y perceptrón multicapa) y selecciona, entre los modelos que no
   superan una tolerancia de predicciones fuera de los límites físicos
   (evaluada sobre todo el periodo de aplicación y no solo sobre el conjunto
   de prueba), el de menor RMSE en prueba.

3. **Verificación.** Vuelve a aplicar el módulo de control de calidad, con los
   mismos umbrales, sobre la serie reconstruida.

## Estructura

```
.
├── app/
│   ├── app_1.py                 Aplicación Streamlit
│   ├── requirements_app_1.txt   Dependencias solo de la app
│   └── .streamlit/config.toml
├── src/qcsolar/
│   ├── config.py                Lectura de config.yaml
│   ├── io.py                    Carga y normalización de series
│   ├── solar.py                 Geometría solar y cielo despejado (pvlib)
│   ├── metrics.py               RMSE, MAE, MBE, R², r
│   ├── qc/
│   │   ├── physical.py          Pruebas físicas (nivel 1)
│   │   ├── expert.py            Pruebas expertas y de serie temporal
│   │   └── usability.py         Dato utilizable y regla de islas
│   ├── siteadapt/
│   │   ├── prepare.py           Emparejamiento y conjuntos de datos
│   │   ├── models.py            Los nueve modelos comparados
│   │   └── evaluate.py          Métricas y criterio de selección
│   ├── report.py                Tablas de resultados
│   └── viz.py                   Figuras de diagnóstico
├── config.yaml                  Toda la parametrización del proyecto
├── pyproject.toml
└── verificar_pieza*.py          Scripts que ejecutan el pipeline por etapas
```

Ningún valor numérico relevante está escrito en el código: las coordenadas del
sitio, las rutas de los archivos, los umbrales de cada prueba, la
correspondencia entre cada fuente terrestre y su fuente satelital y las
ventanas de exclusión viven en `config.yaml`.

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/<usuario>/qc-siteadapt-solar.git
cd qc-siteadapt-solar

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e .            # el paquete qcsolar y sus dependencias
pip install -e ".[app]"     # además, Streamlit para la aplicación web
```

## Datos

**Este repositorio no incluye datos.** Las series usadas en el trabajo
provienen de dos orígenes con condiciones distintas:

- **Solcast** es un producto comercial y sus términos de uso no permiten
  redistribuir los datos. Se obtienen directamente en
  [solcast.com](https://solcast.com), que ofrece acceso gratuito para uso
  académico.
- **Las series terrestres** de la Universidad de Nariño (irradiancia derivada
  de un inversor fotovoltaico y estación meteorológica del campus Torobajo)
  se pueden solicitar al autor.

Para ejecutar el pipeline, los archivos deben ubicarse en `data/raw/` con los
nombres declarados en el bloque `datos` de `config.yaml`, o bien modificar esas
rutas en el archivo de configuración.

## Uso

### Pipeline completo

Los scripts se ejecutan en el orden de su numeración, desde la raíz del
proyecto. Cada uno deja sus resultados en `data/processed/` y `outputs/`:

```bash
python preparar_udenar_ghi.py              # deriva la irradiancia del inversor
python verificar_pieza1_solar.py           # geometría solar
python verificar_pieza2_qc_fisico.py       # pruebas físicas
python verificar_pieza3_qc_experto.py      # pruebas expertas
python verificar_pieza4_usabilidad.py      # dato utilizable y regla de islas
python verificar_pieza5b_emparejamiento_4estaciones.py
python verificar_pieza6c_modelos_4estaciones.py
python verificar_pieza7b_reconstruccion_4estaciones.py
```

Los nombres de los scripts conservan una numeración por paso del pipeline;
todos operan sobre las fuentes declaradas en `config.yaml`.

### Aplicación web

```bash
streamlit run app/app_1.py
```

Las pestañas de resultados leen lo que el pipeline haya dejado en `outputs/`,
así que estarán vacías en una instalación nueva hasta que se ejecuten los
scripts anteriores. La pestaña **Nuevo dataset** es independiente: corre el
pipeline completo en vivo sobre un par de archivos que el usuario cargue desde
su disco, sin necesidad de los datos del proyecto.

## Cómo citar

Si este código le resulta útil, puede citar el trabajo asociado:

> J. A. Chamorro Enríquez, "Análisis del desempeño de la aplicación de técnicas
> de control de calidad y site-adaptation en datos de recurso solar", Trabajo de
> grado, Maestría en Ingeniería Electrónica, Universidad de Nariño, Pasto,
> Colombia, 2026.

## Referencias del método

Los umbrales y las pruebas de control de calidad siguen a Forstinger et al.
(2021), *Expert quality control of solar radiation ground data sets*, Solar
Energy 236, que recoge los límites de la red BSRN publicados por Long y Dutton
(2002). La categorización de las técnicas de *site-adaptation* sigue a Polo et
al. (2016), Solar Energy 132. La lista completa de referencias está en el
artículo.

## Uso de herramientas de asistencia

En el desarrollo del software se usaron herramientas de inteligencia
artificial como asistente de programación, para escribir, depurar y mejorar
partes del código. La metodología, la selección de las pruebas de control de
calidad y de los modelos, el criterio de selección, el análisis de los
resultados y las decisiones técnicas son del autor, que revisó, ejecutó y
validó todo el código de este repositorio.

## Licencia

MIT. Ver [LICENSE](LICENSE).

## Autor

Jorge Andrés Chamorro Enríquez — jordres78@udenar.edu.co
Asesor: Javier Revelo Fuelagán
Departamento de Electrónica, Universidad de Nariño.
