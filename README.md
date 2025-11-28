# tti-lab

Repositorio base para el laboratorio de simulación "tti-lab".

Rápido inicio:
1. Crear el entorno conda:
   conda env create -f ./env/environment.yml

2. Activar el entorno:
   conda activate tti-lab

3. (Opcional) Instalar pytest:
   conda install -n tti-lab pytest -c conda-forge

4. Ejecutar el notebook `notebooks/00_setup_env.ipynb` en JupyterLab o VS Code.

Estructura resumida:
- `src/tti/` : código fuente (core y modelos).
- `notebooks/` : notebooks exploratorios.
- `tests/` : tests unitarios con `pytest`.
- `docs/` : documentación técnica.