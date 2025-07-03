import cv2
import numpy as np
import os

def load_image(path, grayscale=True):
    """
    Carrega uma imagem do disco.
    Args:
        path (str): Caminho da imagem.
        grayscale (bool): Se True, carrega em escala de cinza.
    Returns:
        np.ndarray: Imagem carregada.
    """
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    return img

def binarize_image(img, threshold=127):
    """
    Binariza a imagem usando um limiar fixo.
    Args:
        img (np.ndarray): Imagem em escala de cinza.
        threshold (int): Valor do limiar.
    Returns:
        np.ndarray: Imagem binarizada.
    """
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary

def resize_image(img, size=(155, 220)):
    """
    Redimensiona a imagem para o tamanho padrão.
    Args:
        img (np.ndarray): Imagem de entrada.
        size (tuple): Novo tamanho (altura, largura).
    Returns:
        np.ndarray: Imagem redimensionada.
    """
    return cv2.resize(img, (size[1], size[0]))

def normalize_image(img):
    """
    Normaliza os pixels para o intervalo [0, 1].
    Args:
        img (np.ndarray): Imagem de entrada.
    Returns:
        np.ndarray: Imagem normalizada.
    """
    return img.astype('float32') / 255.0

def preprocess_image(path):
    """
    Executa todo o pipeline de pré-processamento em uma imagem.
    Args:
        path (str): Caminho da imagem.
    Returns:
        np.ndarray: Imagem pronta para o modelo.
    """
    img = load_image(path)
    img = binarize_image(img)
    img = resize_image(img)
    img = normalize_image(img)
    return img

def load_images_from_folder(folder):
    """
    Carrega e pré-processa todas as imagens de uma pasta.
    Args:
        folder (str): Caminho da pasta.
    Returns:
        list: Lista de imagens pré-processadas.
    """
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(folder, filename)
            img = preprocess_image(path)
            images.append(img)
    return images
