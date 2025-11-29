# TTI-Lab — Overview Técnico

Propósito:
Implementar un laboratorio modular para simulaciones computacionales relacionadas con modelos discretos y de automátas celulares. 

Módulos principales:
- `src/tti/core/fields.py`: estructuras de datos para campos discretos (2D).
- `src/tti/models/cellular_automata.py`: funciones y kernels para pasos de autómatas celulares.
- `notebooks/`: experimentos reproducibles y demostraciones.
- `tests/`: pruebas unitarias básicas.

Objetivo técnico:
Proveer una base clara y testeable para iterar modelos CA y otros modelos discretos, con enfoque en reproducibilidad (entorno conda) y pruebas automáticas.

Este repositorio proporciona librerías y herramientas para simular procesos informacionales en campos discretos y continuos. 
Módulos principales:
- `core.probability`: vectores de probabilidad, normalización, transformaciones.
- `core.information`: entropía, información mutua, divergencia KL, etc.
- `core.operators`: operadores de evolución (matrices de transición, kernels).
- `simulations.dynamics`: integradores y simulaciones temporales.
- `visualizations.plots`: funciones para visualización reproducible.
- `math.tensors`: utilidades para tensores y manipulaciones multi-índice.

Estrategia:
- Mantener funciones pequeñas y testeables.
- Vectorizar operaciones para rendimiento.
- Soporte opcional para JAX/PyTorch si se requiere autograd/GPU.

