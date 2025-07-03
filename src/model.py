import tensorflow as tf
from tensorflow.keras import layers, Model, Input


def build_base_network(input_shape):
    """
    Cria a base convolucional para extrair características das assinaturas.
    Args:
        input_shape (tuple): Formato da imagem de entrada (altura, largura, canais).
    Returns:
        keras.Model: Modelo base.
    """
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(64, activation='relu')(x)
    return Model(inputs, outputs, name="base_network")


def euclidean_distance(vectors):
    """
    Calcula a distância euclidiana entre dois vetores.
    Args:
        vectors (list): Lista com dois tensores.
    Returns:
        Tensor: Distância euclidiana.
    """
    x, y = vectors
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))


def build_siamese_network(input_shape):
    """
    Monta a arquitetura da Rede Neural Siamesa.
    Args:
        input_shape (tuple): Formato da imagem de entrada.
    Returns:
        keras.Model: Modelo siamesa completo.
    """
    base_network = build_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])

    model = Model([input_a, input_b], distance, name="siamese_network")
    return model


def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Função de custo Contrastive Loss.
    Args:
        y_true (Tensor): Rótulos (0: genuína, 1: falsa).
        y_pred (Tensor): Distância prevista.
        margin (float): Margem para separação.
    Returns:
        Tensor: Valor da perda.
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
