import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_preprocessing import preprocess_image
from model import build_siamese_network, euclidean_distance

# Parâmetros
INPUT_SHAPE = (155, 220, 1)

# Função para carregar pares de teste para avaliação
def load_test_pairs(test_dir):
    '''
    Carrega pares de imagens de teste e rótulos.
    Espera subpastas para cada autor.
    '''
    images = []
    labels = []
    authors = os.listdir(test_dir)
    for author in authors:
        author_dir = os.path.join(test_dir, author)
        if os.path.isdir(author_dir):
            author_imgs = [os.path.join(author_dir, f) for f in os.listdir(author_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            for i in range(len(author_imgs)):
                for j in range(i+1, len(author_imgs)):
                    img_a = preprocess_image(author_imgs[i])
                    img_b = preprocess_image(author_imgs[j])
                    if img_a.ndim == 2:
                        img_a = np.expand_dims(img_a, axis=-1)
                    if img_b.ndim == 2:
                        img_b = np.expand_dims(img_b, axis=-1)
                    images.append((img_a, img_b))
                    labels.append(1)  # mesmo autor
    # Pares negativos (autores diferentes)
    for i in range(len(authors)):
        for j in range(i+1, len(authors)):
            author_a_imgs = [os.path.join(test_dir, authors[i], f) for f in os.listdir(os.path.join(test_dir, authors[i]))
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            author_b_imgs = [os.path.join(test_dir, authors[j], f) for f in os.listdir(os.path.join(test_dir, authors[j]))
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if author_a_imgs and author_b_imgs:
                img_a = preprocess_image(author_a_imgs[0])
                img_b = preprocess_image(author_b_imgs[0])
                if img_a.ndim == 2:
                    img_a = np.expand_dims(img_a, axis=-1)
                if img_b.ndim == 2:
                    img_b = np.expand_dims(img_b, axis=-1)
                images.append((img_a, img_b))
                labels.append(0)  # autores diferentes
    X_a = np.array([pair[0] for pair in images])
    X_b = np.array([pair[1] for pair in images])
    y = np.array(labels)
    return X_a, X_b, y

if __name__ == "__main__":
    # Carregar pares de teste
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")  # ajuste se necessário
    X_a, X_b, y_true = load_test_pairs(test_dir)
    print(f"Total de pares de teste: {len(y_true)}")
    # Carregar modelo treinado
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "siamese_signature_model.h5")
    model = tf.keras.models.load_model(model_path, custom_objects={'euclidean_distance': euclidean_distance}, compile=False)
    # Prever distâncias
    y_pred_dist = model.predict([X_a, X_b])
    # Definir limiar para decisão (ajuste conforme necessário)
    threshold = 0.5
    y_pred = (y_pred_dist < threshold).astype(int).flatten()
    # Avaliar métricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Matriz de Confusão:")
    print(cm)
