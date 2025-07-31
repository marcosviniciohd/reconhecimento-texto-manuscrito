#!/usr/bin/env python3
"""
Módulo do modelo de Rede Neural Siamesa para verificação de assinaturas.
Contém a arquitetura da rede, funções de perda e métricas.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras import backend as K
import numpy as np


def euclidean_distance(vectors):
    """
    Calcula a distância euclidiana entre dois vetores.
    
    Args:
        vectors: Lista com dois tensores [vector1, vector2]
    
    Returns:
        tensor: Distância euclidiana entre os vetores
    """
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def contrastive_loss(y_true, y_pred):
    """
    Função de perda contrastiva para treinamento da rede siamesa.
    
    Args:
        y_true: Labels verdadeiros (0 = mesma pessoa, 1 = pessoas diferentes)
        y_pred: Distâncias preditas pelo modelo
    
    Returns:
        tensor: Valor da perda contrastiva
    """
    margin = 1.0
    
    # y_true = 0 para mesma pessoa, 1 para pessoas diferentes
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    
    return K.mean((1 - y_true) * square_pred + y_true * margin_square)


def build_base_network(input_shape):
    """
    Constrói a rede base (CNN) para extração de features.
    
    Args:
        input_shape: Formato da entrada (altura, largura, canais)
    
    Returns:
        Model: Modelo da rede base
    """
    input_layer = Input(shape=input_shape)
    
    # Primeira camada convolucional
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    
    # Segunda camada convolucional
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Terceira camada convolucional
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Quarta camada convolucional
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten e camadas densas
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(128, activation='relu')(x)  # Feature vector de 128 dimensões
    
    return Model(input_layer, output)


def build_siamese_network(input_shape):
    """
    Constrói a rede siamesa completa.
    
    Args:
        input_shape: Formato da entrada (altura, largura, canais)
    
    Returns:
        Model: Modelo da rede siamesa
    """
    # Construir rede base
    base_network = build_base_network(input_shape)
    
    # Definir entradas para as duas imagens
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    # Processar ambas as imagens com a mesma rede base
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Calcular distância euclidiana entre os features
    distance = Lambda(euclidean_distance)([processed_a, processed_b])
    
    # Criar modelo final
    model = Model([input_a, input_b], distance)
    
    return model


def create_pairs(images, labels):
    """
    Cria pares de imagens para treinamento da rede siamesa.
    
    Args:
        images: Array com as imagens
        labels: Array com os labels (pessoa1, pessoa2, etc.)
    
    Returns:
        tuple: (pares_imagens, labels_pares)
    """
    pairs = []
    labels_pairs = []
    
    n_classes = len(np.unique(labels))
    
    # Criar índices por classe
    class_indices = {}
    for i, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    # Criar pares positivos (mesma pessoa)
    for label in class_indices:
        indices = class_indices[label]
        # Criar todas as combinações possíveis dentro da classe
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pairs.append([images[indices[i]], images[indices[j]]])
                labels_pairs.append(0)  # 0 = mesma pessoa
    
    # Criar pares negativos (pessoas diferentes)
    labels_list = list(class_indices.keys())
    for i, label1 in enumerate(labels_list):
        for j, label2 in enumerate(labels_list):
            if i < j:  # Evitar duplicatas
                indices1 = class_indices[label1]
                indices2 = class_indices[label2]
                
                # Criar alguns pares entre classes diferentes
                for idx1 in indices1[:min(2, len(indices1))]:  # Máximo 2 por classe
                    for idx2 in indices2[:min(2, len(indices2))]:
                        pairs.append([images[idx1], images[idx2]])
                        labels_pairs.append(1)  # 1 = pessoas diferentes
    
    return np.array(pairs), np.array(labels_pairs)


def predict_similarity(model, img1, img2):
    """
    Prediz a similaridade entre duas imagens.
    
    Args:
        model: Modelo treinado
        img1: Primeira imagem preprocessada
        img2: Segunda imagem preprocessada
    
    Returns:
        float: Distância entre as imagens (menor = mais similar)
    """
    # Expandir dimensões para batch
    img1_batch = np.expand_dims(img1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    
    # Fazer predição
    distance = model.predict([img1_batch, img2_batch])[0][0]
    
    return float(distance)


def evaluate_threshold(model, test_pairs, test_labels, thresholds=None):
    """
    Avalia diferentes thresholds para determinar o ótimo.
    
    Args:
        model: Modelo treinado
        test_pairs: Pares de teste
        test_labels: Labels de teste
        thresholds: Lista de thresholds para testar
    
    Returns:
        dict: Resultados da avaliação para cada threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.31, 0.05)
    
    # Fazer predições para todos os pares
    predictions = model.predict(test_pairs)
    
    results = {}
    
    for threshold in thresholds:
        # Classificar baseado no threshold
        predicted_labels = (predictions > threshold).astype(int)
        
        # Calcular métricas
        correct = (predicted_labels.flatten() == test_labels).sum()
        accuracy = correct / len(test_labels)
        
        # Calcular precisão, recall e F1 para classe "mesma pessoa" (0)
        true_positives = ((predicted_labels.flatten() == 0) & (test_labels == 0)).sum()
        false_positives = ((predicted_labels.flatten() == 0) & (test_labels == 1)).sum()
        false_negatives = ((predicted_labels.flatten() == 1) & (test_labels == 0)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[threshold] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    return results


def load_model_with_custom_objects(model_path):
    """
    Carrega modelo com objetos customizados.
    
    Args:
        model_path: Caminho para o modelo salvo
    
    Returns:
        Model: Modelo carregado
    """
    custom_objects = {
        'euclidean_distance': euclidean_distance,
        'contrastive_loss': contrastive_loss
    }
    
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
