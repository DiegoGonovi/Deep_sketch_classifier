# Este script divide el conjunto de datos en los conjuntos de train, test y validación.
# Se importan dependencias
import os
from tqdm import *
from sklearn.model_selection import train_test_split

# Se carga el conjunto de datos
path = r'C:\Users\diego\Desktop\MUVA\1º\1ºCuatri\ReconocimientoPatrones\P2_dl\imagenes'

# Se inicializan diccionarios y listas que almacenarán datos sobre las clases y archivos.
clases = {} # Almacenará la cantidad de archivos por clase.
total_files = [] # Lista para almacenar todas las rutas de los archivos.
total_clases = [] # Lista para almacenar las clases correspondientes a cada archivo.

# Se recorre todas las carpetas en la ruta especificada.
for filename in tqdm(os.listdir(path)):
    folder = os.path.join(path, filename) # Se obtiene la ruta completa de la carpeta actual.
    num_files = len(os.listdir(folder)) # Se cuenta el número de archivos en la carpeta.
    clases[filename] = num_files # Se almacena el número de archivos de esta clase en el diccionario.

    # Se recorre todos los archivos dentro de la carpeta actual.
    for file in os.listdir(folder):
        source = os.path.join(folder, file) # S obtiene la ruta completa del archivo.
        total_files.append(source) # Se agrega la ruta del archivo a la lista total_files.
        total_clases.append(filename) # Se agrega el nombre de la clase a la lista total_clases.

# Se muestra la distribución de las clases (cantidad de archivos por clase).
print('Distribución de clases en el dataset:')
print()
for i in clases:
    print(i + ': ', clases[i])

print()
print('Total de imágenes en el dataset:', len(total_files))
print('Total de clases en el dataset:', len(clases))
print()

# Separación estratificada del conjunto de datos en train, validación y test.

# El primer train_test_split divide en entrenamiento (80%) y una combinación de validación/prueba (20%).
train, val_, _, y_val = train_test_split(total_files, total_clases, test_size=0.2,
                                        stratify=total_clases, random_state=42)

# El segundo train_test_split divide la parte de validación/prueba en dos partes iguales (10% para cada una).
val, test = train_test_split(val_, test_size=0.5, stratify=y_val, random_state=42)
print('Separación estratificada:')
print('Train:', len(train))
print('Val:', len(val))
print('Test:', len(test))
print()

# Se define las rutas donde se guardarán las listas de rutas de los archivos divididos por conjunto.
train_path = r'C:\Users\diego\Desktop\MUVA\1º\1ºCuatri\ReconocimientoPatrones\P2_dl\train.txt'
val_path = r'C:\Users\diego\Desktop\MUVA\1º\1ºCuatri\ReconocimientoPatrones\P2_dl\validation.txt'
test_path = r'C:\Users\diego\Desktop\MUVA\1º\1ºCuatri\ReconocimientoPatrones\P2_dl\test.txt'

# Se crea y escribe el archivo 'train.txt' con las rutas de los archivos de entrenamiento.
with open(train_path, 'w+',encoding='utf-8') as f:
    for i in train:
        f.write(i + '\n')

# Se crea y escribe el archivo 'validation.txt' con las rutas de los archivos de validación.
with open(val_path, 'w+',encoding='utf-8') as f:
    for i in val:
        f.write(i + '\n')

# Se crea y escribe el archivo 'test.txt' con las rutas de los archivos de prueba.
with open(test_path, 'w+',encoding='utf-8') as f:
    for i in test:
        f.write(i + '\n')

print(f"Archivo 'train.txt' creado en: {train_path}")
print(f"Archivo 'validation.txt' creado en: {val_path}")
print(f"Archivo 'test.txt' creado en: {test_path}")
