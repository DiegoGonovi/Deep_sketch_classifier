# Este es el script correspondiente al producto final a entregar al cliente
# Se importan las dependencias
from ultralytics import YOLO
import shutil
import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import pandas as pd

# Función principal para la inferencia con datos nuevos (test).
def main():
    # Se cargan los datos y el modelo preentrenado
    dataset_path = r'C:\Users\diego\Desktop\MUVA\1º\1ºCuatri\ReconocimientoPatrones\P2_dl'
    model = YOLO(r'C:\Users\diego\Desktop\MUVA\1º\1ºCuatri\ReconocimientoPatrones\P2_dl\cls_model.pt')
    etiquetas_path = r'C:\Users\diego\Desktop\MUVA\1º\1ºCuatri\ReconocimientoPatrones\P2_dl\etiquetas_estimadas.txt'

    # Se lee el archivo 'test.txt' para obtener las rutas de las imágenes del conjunto de prueba.
    with open(dataset_path + '/test.txt', 'r', encoding='utf-8') as file:
        lst_text_images = [line.strip() for line in file]

    # Hacer la inferencia usando el modelo de Yolo
    results = model(lst_text_images, max_det=1, device=0, save=False, show_labels=False, show_conf=False,
                    show_boxes=False)
    # Se inicializan listas para guardar las etiquetas reales (ground truth) e inferidas (predicciones)
    gt = []
    predict = []

    # Se crea un archivo para guardar las etiquetas predichas asociadas a cada ejemplo
    with open(etiquetas_path, 'w+', encoding='utf-8') as label:
        for result in results:
            image_path = result.path # Se obtiene la ruta de la imagen procesada.
            idx = result.probs.top1 # Se obtiene el índice de la clase con mayor probabilidad.
            clas = result.names[idx] # Se obtiene el nombre de la clase predicha.

            # Se escribe la ruta de la imagen y su clase predicha en el archivo de etiquetas.
            label.write(image_path + ',' + str(clas) + '\n')

            # Se extraen la etiqueta real (nombre de la carpeta) y la etiqueta predicha.
            gt.append(image_path.split('\\')[-2]) # Se obtiene la etiqueta real desde la ruta de la imagen.
            predict.append(clas) # Se guarda la etiqueta predicha.

    # Se llama a la función para calcular y guardar la matriz de confusión.
    conf_matrix(dataset_path, gt, predict)
    # Se llama a la función para mostrar las imágenes con las etiquetas predichas.
    show_inference_image(lst_text_images, predict)

# Función para calcular y guardar la matriz de confusión.
def conf_matrix(ropt_path, ground_t, predictions):
    # Se obtienen todas las clases únicas presentes en las etiquetas reales y predichas.
    all_clases = np.unique(ground_t + predictions)
    # Se calcula la matriz de confusión.
    matriz_confusion = confusion_matrix(ground_t, predictions, labels=all_clases)

    # Se convierte la matriz de confusión en un DataFrame para facilitar su exportación.
    df_matriz_confusion = pd.DataFrame(matriz_confusion, index=[f"{clase}" for clase in all_clases],
        columns=[f"{clase}" for clase in all_clases])

    # Se guarda la matriz de confusión en un archivo de texto.
    with open(ropt_path + "\matriz_confusion.txt", "w+", encoding='utf-8') as file:
        file.write(df_matriz_confusion.to_string())

# Función para mostrar las imágenes con las etiquetas predichas.
def show_inference_image(path_images, predictions):
    # Se recorren las imágenes y sus etiquetas predichas.
    for i, j in zip(path_images, predictions):
        image = cv2.imread(i) # Se lee la imagen desde su ruta.

        # Se define el tamaño de la ventana para mostrar la imagen.
        screen_width = 800
        screen_height = 600

        # Se escala la imagen para ajustarla a la ventana.
        height, width = image.shape[:2]
        scale = min(screen_width / width, screen_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))

        # Se configura el texto para mostrar la etiqueta predicha sobre la imagen.
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 1
        color_text = (255, 255, 255)
        color_background = (0, 0, 0)
        position = (7, 45)

        # Se agrega el texto con la etiqueta predicha a la imagen.
        text_size = cv2.getTextSize(j, font, font_scale, thickness)[0]
        text_width, text_height = text_size

        x, y = position
        rect_start = (x - 7, y - text_height - 5)
        rect_end = (x + text_width + 10, y + 5)

        # Se dibuja un rectángulo detrás del texto.
        cv2.rectangle(resized_image, rect_start, rect_end, color_background, -1)

        cv2.putText(resized_image, j, position, font, font_scale, color_text, thickness)

        # Se muestra la imagen con la etiqueta predicha.
        cv2.imshow('Predict', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Se ejecuta la función principal solo si el script se ejecuta directamente.
if __name__ == '__main__':
    main()

