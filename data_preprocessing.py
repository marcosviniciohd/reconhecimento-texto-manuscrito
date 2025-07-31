#!/usr/bin/env python3
"""
Módulo de preprocessamento de dados para assinaturas manuscritas.
Contém funções para carregar, redimensionar, binarizar e normalizar imagens.
"""

import cv2
import numpy as np
from PIL import Image


def preprocess_image(image_path, target_size=(220, 155)):
    """
    Preprocessa uma imagem de assinatura para o formato esperado pelo modelo.
    
    Args:
        image_path (str): Caminho para a imagem
        target_size (tuple): Tamanho alvo (largura, altura)
    
    Returns:
        np.array: Imagem preprocessada normalizada
    """
    try:
        # Carregar imagem
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        # Binarizar usando threshold OTSU
        _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Redimensionar
        img_resized = cv2.resize(img_binary, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalizar para [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Expandir dimensões para o formato esperado pelo modelo
        img_final = np.expand_dims(img_normalized, axis=-1)
        
        return img_final
        
    except Exception as e:
        print(f"Erro ao preprocessar imagem {image_path}: {e}")
        # Retorna imagem vazia em caso de erro
        empty_img = np.zeros((155, 220, 1), dtype=np.float32)  # (altura, largura, canais)
        return empty_img


def binarize_image(img_array, threshold=None):
    """
    Binariza uma imagem usando threshold OTSU ou valor específico.
    
    Args:
        img_array (np.array): Array da imagem em escala de cinza
        threshold (float, optional): Valor de threshold. Se None, usa OTSU
    
    Returns:
        np.array: Imagem binarizada
    """
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    
    if threshold is None:
        # Usar OTSU para threshold automático
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Usar threshold específico
        _, binary = cv2.threshold(img_array, int(threshold * 255), 255, cv2.THRESH_BINARY_INV)
    
    return binary


def resize_image(img_array, target_size=(220, 155)):
    """
    Redimensiona uma imagem para o tamanho alvo.
    
    Args:
        img_array (np.array): Array da imagem
        target_size (tuple): Tamanho alvo (largura, altura)
    
    Returns:
        np.array: Imagem redimensionada
    """
    return cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(img_array):
    """
    Normaliza uma imagem para o intervalo [0, 1].
    
    Args:
        img_array (np.array): Array da imagem
    
    Returns:
        np.array: Imagem normalizada
    """
    if img_array.dtype == np.uint8:
        return img_array.astype(np.float32) / 255.0
    return img_array.astype(np.float32)


def preprocess_streamlit_image(uploaded_file, target_size=(220, 155)):
    """
    Preprocessa uma imagem carregada via Streamlit.
    
    Args:
        uploaded_file: Objeto UploadedFile do Streamlit
        target_size (tuple): Tamanho alvo (largura, altura)
    
    Returns:
        np.array: Imagem preprocessada
    """
    try:
        # Converter para PIL Image
        pil_image = Image.open(uploaded_file)
        
        # Converter para escala de cinza se necessário
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        
        # Converter para numpy array
        img_array = np.array(pil_image)
        
        # Binarizar
        img_binary = binarize_image(img_array)
        
        # Redimensionar
        img_resized = resize_image(img_binary, target_size)
        
        # Normalizar
        img_normalized = normalize_image(img_resized)
        
        # Expandir dimensões
        img_final = np.expand_dims(img_normalized, axis=-1)
        
        return img_final
        
    except Exception as e:
        print(f"Erro ao preprocessar imagem do Streamlit: {e}")
        # Retorna imagem vazia em caso de erro
        empty_img = np.zeros((155, 220, 1), dtype=np.float32)  # (altura, largura, canais)
        return empty_img


def enhance_phone_image(img_array):
    """
    Aplica melhorias específicas para imagens capturadas por telefone.
    
    Args:
        img_array (np.array): Array da imagem
    
    Returns:
        np.array: Imagem melhorada
    """
    try:
        # Garantir que é uint8
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        # 1. Redução de ruído
        denoised = cv2.medianBlur(img_array, 3)
        
        # 2. Equalização de histograma
        equalized = cv2.equalizeHist(denoised)
        
        # 3. Threshold automático OTSU
        _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return binary
        
    except Exception as e:
        print(f"Erro ao melhorar imagem de telefone: {e}")
        return img_array


def load_and_preprocess_batch(image_paths, target_size=(220, 155)):
    """
    Carrega e preprocessa um lote de imagens.
    
    Args:
        image_paths (list): Lista de caminhos para as imagens
        target_size (tuple): Tamanho alvo
    
    Returns:
        np.array: Array com todas as imagens preprocessadas
    """
    images = []
    for path in image_paths:
        img = preprocess_image(path, target_size)
        images.append(img)
    
    return np.array(images)
