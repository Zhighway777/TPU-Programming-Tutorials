

# awesome tpu tutorial

![GitHub stars](https://img.shields.io/github/stars/Zhighway777/TPU-Programming-Tutorials?style=social)
![GitHub forks](https://img.shields.io/github/forks/Zhighway777/TPU-Programming-Tutorials?style=social)
![GitHub issues](https://img.shields.io/github/issues/Zhighway777/TPU-Programming-Tutorials)
![GitHub license](https://img.shields.io/github/license/Zhighway777/TPU-Programming-Tutorials)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

[English](#english) | [中文](#chinese)

---

## <a name="english"></a>Inglés

### 📚 Acerca de este repositorio

> 🎓 **Enfoque**: Programación de TPU consciente de la arquitectura  
> Este repositorio hace hincapié en **cómo interactúan la arquitectura de hardware de TPU, la compilación XLA/HLO y el diseño a nivel de kernel**, en lugar de limitarse únicamente al uso de APIs.


Un repositorio de tutoriales completo que cubre la programación y la arquitectura de **TPU (Tensor Processing Unit)**. Este repositorio proporciona materiales educativos, tutoriales prácticos, ejemplos de código y recursos para desarrolladores, investigadores y estudiantes interesados en aprender la programación de TPU.

### 🔗 Recursos

#### Artículos
- [Seguimiento de la última arquitectura de Google TPU](docs/architecture/Google's_latest_TPU_tracking.en.md)
- [De TPUv3 a TPUv4i](docs/architecture/Google-TPUv4-Architecture-Deep-Dive.en.md)
- [Antecedentes de TPUv1~v3 Revelados](docs/architecture/TPUv1~v3_revealed.en.md)

#### Blogs
- [Una mirada profunda a la primera unidad de procesamiento de tensor (TPU) de Google](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)
- [Google potencia las tareas de aprendizaje automático con un chip personalizado](https://cloud.google.com/blog/products/ai-machine-learning/google-supercharges-machine-learning-tasks-with-custom-chip)
- [Análisis profundo de TPU](https://henryhmko.github.io/posts/tpu/tpu.html)
- [Arquitectura de TPU de Google: Guía completa de 7 generaciones](https://www.introl.io/blog/google-tpu-architecture-complete-guide-7-generations)

#### Diapositivas
- [Chips de entrenamiento de Google revelados: TPUv2 y TPUv3](https://www.hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf)
- [Diez lecciones: 4 generaciones de TPU](https://www.cs.ucla.edu/wp-content/uploads/cs/PATTERSON-10-Lessons-4-TPU-gens-CO2e-45-minutes.pdf)
- [Análisis de rendimiento de un centro de datos TPU](https://andrewt0301.github.io/hse-acos-course/part1ca/16_TPU/NAETPUslides5Apr17v2.pdf)
- [Diez lecciones de tres generaciones que moldearon el producto industrial TPUv4i de Google (EPFL CS723)](https://parsa.epfl.ch/course-info/cs723/lectures/hw_accel.pdf)
- [Una supercomputadora de aprendizaje automático con una interconexión reconfigurable ópticamente y soporte de incrustaciones](https://hc2023.hotchips.org/assets/program/conference/day2/ML%20training/HC2023.Session5.ML_Training.Google.Norm_Jouppi.Andy_Swing.Final_2023-08-25.pdf)

#### Artículos científicos
- [Análisis de rendimiento de una unidad de procesamiento de tensor en un centro de datos](https://arxiv.org/pdf/1704.04760)
- [El proceso de diseño de los chips de entrenamiento de Google: TPUv2 y TPUv3](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf)
- [El proceso de diseño de los chips de entrenamiento de Google: TPUv2 y TPUv3 (IEEE)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9351692)
- [Diez lecciones de tres generaciones que moldearon el producto industrial TPUv4i de Google](https://ieeexplore.ieee.org/document/9499913)
- [TPU v4: Una supercomputadora reconfigurable ópticamente para aprendizaje automático con soporte de hardware para incrustaciones](https://arxiv.org/pdf/2304.01433)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Unidades de procesamiento de tensor (TPU): Un análisis técnico y su impacto en la inteligencia artificial](https://tech4future.info/wp-content/uploads/2024/11/Tensor-Processing-Units-TPU-Paper-ENG.pdf)

#### Documentos
- [Documentación de TPU en Google Cloud](https://cloud.google.com/tpu/docs)
- [Guía de TPU para TensorFlow](https://www.tensorflow.org/guide/tpu)
- [Documentación de PyTorch XLA](https://pytorch.org/xla/)
- [JAX en TPU](https://jax.readthedocs.io/)
- [Pallas: un lenguaje de kernel para JAX](https://docs.jax.dev/en/latest/pallas/index.html)
- [Cómo escalar tu modelo | Google DeepMind](https://jax-ml.github.io/scaling-book/)
- [Costo total de propiedad - Wikipedia](https://en.wikipedia.org/wiki/Total_cost_of_ownership)

#### GitHub
- [TPU Starter](https://github.com/ayaka14732/tpu-starter/tree/main)

### 🎯 Qué aprenderás

- **Arquitectura de TPU**: Comprensión del diseño de hardware de TPU, componentes y características de rendimiento
- **Programación de TPU**: Modelos de programación, APIs y marcos para el desarrollo de TPU
- **Técnicas de optimización**: Mejores prácticas para optimizar el rendimiento de TPU
- **Ejemplos prácticos**: Aplicaciones y casos de uso del mundo real
- **Cloud TPU y Edge TPU**: Trabajo con dispositivos Google Cloud TPU y Edge TPU

### 📖 Tabla de contenidos

- [Recursos](#resources)
- [Primeros pasos](#getting-started)
- [Estructura del repositorio](#repository-structure)
- [Tutoriales](#tutorials)
- [Documentación](#documentation)
- [Ejemplos](#examples)
- [Contribuir](#contributing)
- [Licencia](#license)
- [Comunidad](#community)

### 🚀 Primeros pasos

#### Requisitos previos

- Conocimientos básicos de conceptos de aprendizajeaprendizaje automático y aprendizaje profundo
- Familiaridad con la programación en Python
- Conocimiento de TensorFlow o PyTorch (recomendado)

#### Inicio rápido

```bash
# Clonar el repositorio
git clone https://github.com/Zhighway777/awesome-tpu-tutorial.git

# Navegar al repositorio
cd awesome-tpu-tutorial

# Explorar tutoriales
cd tutorials/
```

### 📁 Estructura del repositorio

```
awesome-tpu-tutorial/
├── README.md                    # Este archivo
├── CONTRIBUTING.md              # Guías de contribución
├── LICENSE                      # Información de licencia
├── CODE_OF_CONDUCT.md          # Pautas de la comunidad
├── docs/                        # Documentación
│   ├── architecture/            # Documentación de arquitectura de TPU
│   ├── programming-guides/      # Guías de programación
│   └── api-reference/           # Referencias de API
├── tutorials/                   # Tutoriales paso a paso
│   ├── beginner/               # Tutoriales para principiantes
│   ├── intermediate/           # Tutoriales intermedios
│   └── advanced/               # Tutoriales avanzados
├── examples/                    # Ejemplos de código
│   ├── tensorflow/             # Ejemplos de TensorFlow
│   ├── pytorch/                # Ejemplos de PyTorch
│   └── jax/                    # Ejemplos de JAX
└── resources/                   # Recursos adicionales
    ├── papers/                 # Artículos de investigación
    ├── presentations/          # Diapositivas y presentaciones
    └── references/             # Referencias externas
```

### 📝 Tutoriales

#### Nivel principiante
- Introducción a TPU y sus ventajas
- Configuración del entorno de desarrollo de TPU
- Tu primer programa en TPU
- Operaciones básicas de tensor en TPU

#### Nivel intermedio
- Gestión de memoria en TPU
- Optimización de la canalización de datos
- Parallelismo de modelos en TPU
- Entrenamiento de redes neuronales en TPU

#### Nivel avanzado
- Kernels personalizados de TPU
- Perfilado y optimización del rendimiento
- Entrenamiento distribuido a gran escala
- Investigación de TPU y técnicas de vanguardia

### 📚 Documentación

- **[Guía de arquitectura de TPU](docs/architecture/)**: Análisis profundo del diseño de hardware de TPU
- **[Guías de programación](docs/programming-guides/)**: Tutoriales de programación integra
- **[Referencia de API](docs/api-reference/)**: Documentación detallada de la API

### 💡 Ejemplos

Explora nuestra colección de ejemplos prácticos:
- Modelos de clasificación de imágenes
- Procesamiento del lenguaje natural (NLP)
- Sistemas de recomendación
- Aprendizaje por refuerzo
- Bucles de entrenamiento personalizados

### 🤝 Contribuir

¡Aceptamos contribuciones! Consulta nuestra [Guía de contribución](CONTRIBUTING.md) para detalles sobre:
- Cómo presentarEnviar incidencias (issues)
- Cómo proponer nuevos tutoriales
- Pautas de estilo de código
- Proceso de solicitudesPull Request

### 📄 Licencia

Este proyecto está licenciado bajo la Licencia Apache 2.0. Consulta el archivo [LICENSE](LICENSE) para más detalles.

### 🌟 Agradecimientos

Un agradecimiento especial a todos los colaboradores y a la comunidad de TPU por sus valiosos aportes y apoyo.

---

## <a name="chinese"></a>Chino

### 📚 Acerca del repositorio

> 🎓 **Enfoque del repositorio**: Programación de TPU desde una perspectiva de arquitectura y compilador  
> Este repositorio no solo presenta el uso de la API de TPU, sino que se centra en la **relación entre la arquitectura de hardware de TPU, el flujo de compilación XLA/HLO y el diseño/modelado de rendimiento de kernels**.

Este es un repositorio de tutoriales de programación y arquitectura de TPU (Tensor Processing Unit). Proporciona materiales educativos, tutoriales prácticos, ejemplos de código y recursos de aprendizaje para desarrolladores, investigadores y estudiantes.

### 🔗 Recursos
#### Mis artículos
- [Información más reciente sobre la arquitectura de TPU de Google](docs/architecture/Google's_latest_TPU_tracking.md)
- [De TPUv3 a TPUv4i](docs/architecture/Google-TPUv4-Architecture-Deep-Dive.md)
- [Revisión de la arquitectura de TPUv1~v3](docs/architecture/TPUv1~v3_revealed.md)


#### Blogs técnicos
> Se recomienda primero al público de habla china la serie sobre la evolución de la arquitectura TPU de Google dentro de AI Infra del profesor ZOMI.
- [Historia y evolución de la arquitectura de TPU de Google | ZOMI](https://infrasys-ai.github.io/aisystem-docs/02Hardware05Abroad/04TPUIntrol.html)
- [Tutorial de uso de TPU](https://shizhediao.github.io/TPU-Tutorial/)
- [Análisis profundo de TPU (TPU Deep Dive)](https://henryhmko.github.io/posts/tpu/tpu.html)
- [SemiAnalysis: Análisis profundo de TPU -- Google desafía el "Imperio de NVIDIA"](https://wallstreetcn.com/articles/3760377)
- [Arquitectura de TPU: Guía completa de las 7 generaciones de Google](https://www.introl.io/blog/google-tpu-architecture-complete-guide-7-generations)
- [Una mirada profunda a la primera unidad de procesamiento de tensor (TPU) de Google](https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)
- [Google potencia las tareas de aprendizaje automático con un chip personalizado](https://cloud.google.com/blog/products/ai-machine-learning/google-supercharges-machine-learning-tasks-with-custom-chip)

#### Diapositivas
- [Análisis de rendimiento de un centro de datos TPU](https://andrewt0301.github.io/hse-acos-course/part1ca/16_TPU/NAETPUslides5Apr17v2.pdf)
- [Diez años de evolución de TPU: Diez lecciones de Google](https://www.cs.ucla.edu/wp-content/uploads/cs/PATTERSON-10-Lessons-4-TPU-gens-CO2e-45-minutes.pdf)
- [Chips de entrenamiento de Google revelados: TPUv2 y TPUv3](https://www.hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf)
- [Diez lecciones de tres generaciones que moldearon el producto industrial TPUv4i de Google (EPFL CS723)](https://parsa.epfl.ch/course-info/cs723/lectures/hw_accel.pdf)
- [Una supercomputadora de aprendizaje automático con una interconexión reconfigurable ópticamente y soporte de incrustaciones](https://hc2023.hotchips.org/assets/program/conference/day2/ML%20training/HC2023.Session5.ML_Training.Google.Norm_Jouppi.Andy_Swing.Final_2023-08-25.pdf)

#### Artículos científicos
- [Diez lecciones de tres generaciones que moldearon TPUv4i de Google](https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf)
- [Diez lecciones de tres generaciones que moldearon el producto industrial TPUv4i de Google](https://ieeexplore.ieee.org/document/9499913)
- [Análisis de rendimiento de una unidad de procesamiento de tensor en un centro de datos](https://arxiv.org/pdf/1704.04760)
- [El proceso de diseño de los chips de entrenamiento de Google: TPUv2 y TPUv3](https://gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf)
- [El proceso de diseño de los chips de entrenamiento de Google: TPUv2 y TPUv3 (IEEE)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9351692)
- [TPU v4: Una supercomputadora reconfigurable ópticamente para aprendizaje automático con soporte de hardware para incrustaciones](https://arxiv.org/pdf/2304.01433)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Unidades de procesamiento de tensor (TPU): Un análisis técnico y su impacto en la inteligencia artificial](https://tech4future.info/wp-content/uploads/2024/11/Tensor-Processing-Units-TPU-Paper-ENG.pdf)

#### Documentos técnicos
- [Documentación de TPU en Google Cloud](https://cloud.google.com/tpu/docs)
- [Guía de TPU para TensorFlow](https://www.tensorflow.org/guide/tpu)
- [Documentación de PyTorch XLA](https://pytorch.org/xla/)
- [JAX en TPU](https://jax.readthedocs.io/)
- [Pallas: un lenguaje de kernel para JAX](https://docs.jax.dev/en/latest/pallas/index.html)
- [Cómo escalar tu modelo | Google DeepMind](https://jax-ml.github.io/scaling-book/)
- [Costo total de propiedad - Wikipedia](https://en.wikipedia.org/wiki/Total_cost_of_ownership)

#### GitHub
- [TPU Starter](https://github.com/ayaka14732/tpu-starter/tree/main)

### 🎯 Qué aprenderás

- **Arquitectura de TPU**: Comprensión del diseño de hardware, componentes y características de rendimiento de TPU
- **Programación de TPU**: Modelos de programación, APIs y marcos para el desarrollo en TPU
- **Técnicas de optimización**: Mejores prácticas para optimizar el rendimiento de TPU
- **Ejemplos prácticos**: Aplicaciones y casos de uso del mundo real
- **Cloud TPU y Edge TPU**: Trabajo con dispositivos Google Cloud TPU y Edge TPU

### 📖 Tabla de contenidos

- [Recursos](#resources)
- [Inicio rápido](#getting-started)
- [Estructura del repositorio](#repository-structure)
- [Tutoriales](#tutorials)
- [Documentación](#documentation)
- [Ejemplos](#examples)
- [Contribuir](#contributing)
- [Licencia](#license)
- [Comunidad](#community)

### 🚀 Inicio rápido

#### Requisitos previos

- Conocimientos básicos de aprendizaje automático y aprendizaje profundo
- Familiaridad con la programación en Python
- Conocimiento de TensorFlow o PyTorch (recomendado)

#### Inicio rápido

```bash
# Clonar el repositorio
git clone https://github.com/Zhighway777/TPU-Programming-Tutorials.git

# Entrar en el directorio del repositorio
cd TPU-Programming-Tutorials

# Navegar por los tutoriales
cd tutorials/
```

### 📁 Estructura del repositorio

```
TPU-Programming-Tutorials/
├── README.md                    # Este archivo
├── CONTRIBUTING.md              # GuíaGuías de contribución
├── LICENSE                      # Información de licencia
├── CODE_OF_CONDUCT.md          # Pautas de la comunidad
├── docs/                        # Documentación
│   ├── architecture/            # Documentación de arquitectura de TPU
│   ├── programming-guides/      # Guías de programación
│   └── api-reference/           # Referencias de API
├── tutorials/                   # Tutoriales paso a paso
│   ├── beginner/               # Tutoriales para principiantes
│   ├── intermediate/           # Tutoriales intermedios
│   └── advanced/               # Tutoriales avanzados
├── examples/                    # Ejemplos de código
│   ├── tensorflow/             # Ejemplos de TensorFlow
│   ├── pytorch/                # Ejemplos de PyTorch
│   └── jax/                    # Ejemplos de JAX
└── resources/                   # Recursos adicionales
    ├── papers/                 # Artículos de investigación
    ├── presentations/          # Diapositivas y presentaciones
    └── references/             # Referencias externas
```

### 📝 Tutoriales

#### Principiante
- Introducción a TPU y sus ventajas
- Configuración del entorno de desarrollo de TPU
- Tu primer programa en TPU
- Operaciones básicas de tensor en TPU

#### Intermedio
- Gestión de memoria en TPU
- Optimización de la canalización de datos
- Parallelismo de modelos en TPU
- Entrenamiento de redes neuronales en TPU

#### Avanzado
- Kernels personalizados de TPU
- Perfilado y optimización del rendimiento
- Entrenamiento distribuido a gran escala
- Investigación de TPU y técnicas de vanguardia

### 📚 Documentación

- **[Guía de arquitectura de TPU](docs/architecture/)**: Análisis profundo del diseño de hardware de TPU
- **[Guías de programación](docs/programming-guides/)**: Tutoriales de programación completos
- **[Referencia de API](docs/api-reference/)**: Documentación detallada de la API

### 💡 Ejemplos

Explora nuestra colección de ejemplos prácticos:
- Modelos de clasificación de imágenes
- Procesamiento del lenguaje natural (NLP)
- Sistemas de recomendación
- Aprendizaje por refuerzo
- Bucles de entrenamiento personalizados

### 🤝 Contribuir

¡Aceptamos contribuciones! Consulta nuestra [Guía de contribución](CONTRIBUTING.md) para detalles sobre:
- Cómo enviar incidencias
- Cómo proponer nuevos tutoriales
- Pautas de estilo de código
- Proceso de Pull Request

### 📄 Licencia

Este proyecto utiliza la licencia de código abierto **Apache License 2.0**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## 📊 Estadísticas del repositorio

![GitHub contributors](https://img.shields.io/github/contributors/Zhighway777/TPU-Programming-Tutorials)
![GitHub last commit](https://img.shields.io/github/last-commit/Zhighway777/TPU-Programming-Tutorials)
![GitHub repo size](https://img.shields.io/github/repo-size/Zhighway777/TPU-Programming-Tutorials)

---

**Nota**: Este repositorio se actualiza continuamente con nuevos tutoriales y recursos. ¡Marca ⭐ este repositorio para mantenerte actualizado!
