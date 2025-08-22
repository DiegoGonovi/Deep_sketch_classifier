# 📌Reconocimiento de Bocetos de Objetos Cotidianos mediante Deep Learning
Este proyecto implementa un sistema de aprendizaje profundo para clasificar bocetos dibujados a mano de objetos cotidianos utilizando YOLOv11. El sistema fue entrenado en un subconjunto del dataset de bocetos de SIGGRAPH que contiene 26 categorías de objetos comunes.

## 📂 Estructura del Repositorio
```
/
├── splitter.py                  # Script para división del dataset
├── train.py                     # Script de entrenamiento del modelo
├── finalproduct.py              # Script de inferencia y evaluación
├── train.txt                    # Rutas de imágenes de entrenamiento
├── validation.txt               # Rutas de imágenes de validación
├── test.txt                     # Rutas de imágenes de prueba
├── cls_model.pt                 # Modelo YOLOv11 entrenado
├── etiquetas_estimadas.txt      # Etiquetas predichas para el conjunto de prueba
├── confusion_matrix.txt         # Matriz de confusión
├── memoria_Gonzalez_Pena.pdf    # Reporte del proyecto
└── README.md                    # Presentación del repositorio
```

## 📄 Licencia
Proyecto académico de la Universidad Rey Juan Carlos.
