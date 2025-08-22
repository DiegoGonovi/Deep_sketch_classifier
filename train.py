# Se importan las dependencias
import shutil
from tqdm import *
import os
from ultralytics import YOLO
import torch
import gc


# Función principal que procesa los datos de entrenamiento y validación, carga el modelo y lo entrena.
def main():
    # Se define la ruta base donde se encuentran los datos y se crean carpetas necesarias.
    dataset_path = r'C:\Users\diego\Desktop\MUVA\1º\1ºCuatri\ReconocimientoPatrones\P2_dl'
    root_path = os.path.join(dataset_path, 'dataset')  # Ruta para el dataset procesado.
    os.makedirs(root_path, exist_ok=True)  # Se crea la carpeta principal si no existe.

    train_path = os.path.join(root_path, 'train')  # Ruta para los datos de entrenamiento.
    os.makedirs(train_path, exist_ok=True)  # Se crea la carpeta de entrenamiento si no existe.

    # Se inicializa una lista para guardar las clases procesadas.
    all_clases = []

    # Se lee el archivo 'train.txt' y se copian las imágenes de entrenamiento a sus respectivas carpetas de clase.
    with open(dataset_path + '/train.txt', 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()):  # Se recorre cada línea del archivo.
            clas = line.strip().split('\\')[-2]  # Se extrae el nombre de la clase desde la ruta del archivo.
            clas_path = os.path.join(train_path, clas)  # Se define la ruta donde se almacenará la clase.

            if clas in all_clases:  # Se verifica si la clase ya fue procesada.
                shutil.copy(line.strip(), clas_path)  # Se copia el archivo a la carpeta de la clase existente.

            else:
                os.makedirs(clas_path, exist_ok=True)  # Se crea la carpeta para la nueva clase.
                shutil.copy(line.strip(), clas_path)  # Se copia el archivo a la carpeta recién creada.
                all_clases.append(clas)  # Se agrega la clase a la lista de clases procesadas.

    print('Subconjunto Train Procesado.')
    print()

    # Se crea la carpeta para los datos de validación.
    val_path = os.path.join(root_path, 'val')  # Ruta para los datos de validación.
    os.makedirs(val_path, exist_ok=True)  # Se asegura que la carpeta exista.

    # Se lee el archivo 'validation.txt' y se copian las imágenes de validación a sus respectivas carpetas de clase.
    with open(dataset_path + '/validation.txt', 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines()):  # Se recorre cada línea del archivo.
            clas = line.strip().split('\\')[-2]  # Se extrae el nombre de la clase desde la ruta del archivo.
            clas_path = os.path.join(val_path, clas)  # Se define la ruta donde se almacenará la clase.

            os.makedirs(clas_path, exist_ok=True)  # Se crea la carpeta para la clase si no existe.
            shutil.copy(line.strip(), clas_path)  # Se copia el archivo a la carpeta correspondiente.

    print('Subconjunto Validation Procesado.')

    # Entrenar modelo Yolo
    model = YOLO("yolo11s-cls.pt")  # Se carga el modelo YOLO preentrenado.
    results = model.train(data=r"C:\Users\diego\Desktop\MUVA\1º\1ºCuatri\ReconocimientoPatrones\P2_dl\dataset",
                          epochs=200, imgsz=640, batch=8, patience=50, erasing=0, auto_augment=None, mosaic=0, mixup=0,
                          copy_paste_mode='null', close_mosaic=0, max_det=1, device=0)


# Se ejecuta la función principal solo si este script es ejecutado directamente.
if __name__ == '__main__':
    print('Datos GPU:')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    main()
    torch.cuda.empty_cache()
    gc.collect()