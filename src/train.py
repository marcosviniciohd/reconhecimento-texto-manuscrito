import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_image
from model import build_siamese_network, contrastive_loss

# Parâmetros
INPUT_SHAPE = (155, 220, 1)  # altura, largura, canais
BATCH_SIZE = 16
EPOCHS = 20

# Função para criar pares de imagens e rótulos
def create_pairs(images, labels):
    '''
    Cria pares de imagens (genuína/genuína e genuína/falsa) para treinamento.
    Args:
        images (list): Lista de imagens pré-processadas.
        labels (list): Lista de rótulos (identificador do autor).
    Returns:
        tuple: (pares de imagens A, pares de imagens B, rótulos)
    '''
    pairs_a, pairs_b, pair_labels = [], [], []
    num_classes = len(np.unique(labels))
    label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
    for idx, img_a in enumerate(images):
        label = labels[idx]
        # Par positivo (mesmo autor)
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = np.random.choice(label_to_indices[label])
        img_b = images[pos_idx]
        pairs_a.append(img_a)
        pairs_b.append(img_b)
        pair_labels.append(1)
        # Par negativo (autores diferentes)
        neg_label = np.random.choice(list(set(labels) - set([label])))
        neg_idx = np.random.choice(label_to_indices[neg_label])
        img_b = images[neg_idx]
        pairs_a.append(img_a)
        pairs_b.append(img_b)
        pair_labels.append(0)
    return np.array(pairs_a), np.array(pairs_b), np.array(pair_labels)

# Exemplo de carregamento de dados (ajuste para seu dataset)
def load_dataset(data_dir):
    '''
    Carrega imagens e rótulos do diretório de dados.
    Espera subpastas para cada autor.
    '''
    images = []
    labels = []
    for author in os.listdir(data_dir):
        author_dir = os.path.join(data_dir, author)
        if os.path.isdir(author_dir):
            for fname in os.listdir(author_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(author_dir, fname)
                    img = preprocess_image(img_path)
                    if img.ndim == 2:
                        img = np.expand_dims(img, axis=-1)
                    images.append(img)
                    labels.append(author)
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    # Carregar dataset
    data_dir = "data"  # ajuste se necessário
    images, labels = load_dataset(data_dir)
    print(f"Total de imagens: {len(images)}")
    # Garante que as imagens tenham shape (N, 155, 220, 1)
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    # Criar pares
    pairs_a, pairs_b, pair_labels = create_pairs(images, labels)
    # Dividir em treino e validação
    X_a_train, X_a_val, X_b_train, X_b_val, y_train, y_val = train_test_split(
        pairs_a, pairs_b, pair_labels, test_size=0.2, random_state=42)
    # Construir modelo
    model = build_siamese_network(INPUT_SHAPE)
    model.compile(optimizer='adam', loss=contrastive_loss, metrics=['accuracy'])
    # Treinar
    history = model.fit(
        [X_a_train, X_b_train], y_train,
        validation_data=([X_a_val, X_b_val], y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    # Salvar modelo
    model.save("siamese_signature_model.h5")
    print("Modelo treinado e salvo com sucesso!")
