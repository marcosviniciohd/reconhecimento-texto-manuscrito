#!/usr/bin/env python3
"""
Script para treinar modelo de verificação de assinaturas manuscritas.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

# Adicionar diretório pai ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_preprocessing import preprocess_image
from model import build_siamese_network, contrastive_loss

class ModelTrainer:
    def __init__(self, data_dir="dataset_processado", model_dir="modelos"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.input_shape = (155, 220, 1)
    
    def carregar_dataset(self):
        """Carrega imagens do dataset processado."""
        if not self.data_dir.exists():
            print(f"❌ Dataset não encontrado em {self.data_dir}")
            print("Execute primeiro: python scripts/preparar_dataset.py")
            return None, None
        
        images = []
        labels = []
        
        print("📂 Carregando dataset...")
        
        for pessoa_dir in self.data_dir.iterdir():
            if not pessoa_dir.is_dir():
                continue
            
            pessoa_name = pessoa_dir.name
            pessoa_images = list(pessoa_dir.glob("*.png"))
            
            print(f"   👤 {pessoa_name}: {len(pessoa_images)} imagens")
            
            for img_path in pessoa_images:
                try:
                    img = preprocess_image(str(img_path))
                    if img.ndim == 2:
                        img = np.expand_dims(img, axis=-1)
                    images.append(img)
                    labels.append(pessoa_name)
                except Exception as e:
                    print(f"   ⚠️ Erro ao carregar {img_path}: {e}")
        
        print(f"✅ Dataset carregado: {len(images)} imagens de {len(set(labels))} pessoas")
        return np.array(images), np.array(labels)
    
    def criar_pares(self, images, labels):
        """Cria pares de imagens para treinamento siamês."""
        pairs_a, pairs_b, pair_labels = [], [], []
        unique_labels = list(set(labels))
        
        print("🔗 Criando pares de treinamento...")
        
        # Mapeamento label -> índices
        label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        
        for idx, img_a in enumerate(images):
            label_a = labels[idx]
            
            # Par positivo (mesma pessoa)
            same_indices = [i for i in label_to_indices[label_a] if i != idx]
            if same_indices:
                pos_idx = np.random.choice(same_indices)
                pairs_a.append(img_a)
                pairs_b.append(images[pos_idx])
                pair_labels.append(0)  # 0 = mesma pessoa (distância pequena)
            
            # Par negativo (pessoas diferentes)
            different_labels = [l for l in unique_labels if l != label_a]
            if different_labels:
                neg_label = np.random.choice(different_labels)
                neg_idx = np.random.choice(label_to_indices[neg_label])
                pairs_a.append(img_a)
                pairs_b.append(images[neg_idx])
                pair_labels.append(1)  # 1 = pessoas diferentes (distância grande)
        
        pairs_a = np.array(pairs_a)
        pairs_b = np.array(pairs_b)
        pair_labels = np.array(pair_labels)
        
        print(f"✅ Pares criados: {len(pair_labels)} total")
        print(f"   Pares positivos (mesma pessoa): {np.sum(pair_labels == 0)}")
        print(f"   Pares negativos (pessoas diferentes): {np.sum(pair_labels == 1)}")
        
        return pairs_a, pairs_b, pair_labels
    
    def treinar(self, epochs=25, batch_size=16):
        """Treina o modelo."""
        # Carregar dados
        images, labels = self.carregar_dataset()
        if images is None:
            return False
        
        # Criar pares
        pairs_a, pairs_b, pair_labels = self.criar_pares(images, labels)
        
        # Dividir em treino e validação
        X_a_train, X_a_val, X_b_train, X_b_val, y_train, y_val = train_test_split(
            pairs_a, pairs_b, pair_labels, test_size=0.2, random_state=42, stratify=pair_labels)
        
        print(f"\n📊 Divisão dos dados:")
        print(f"   Treinamento: {len(X_a_train)} pares")
        print(f"   Validação: {len(X_a_val)} pares")
        
        # Construir modelo
        print(f"\n🏗️ Construindo modelo...")
        model = build_siamese_network(self.input_shape)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=contrastive_loss,
            metrics=['accuracy']
        )
        
        print(f"✅ Modelo construído")
        model.summary()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )
        ]
        
        # Treinar
        print(f"\n🚀 Iniciando treinamento...")
        print(f"   Épocas: {epochs}")
        print(f"   Batch size: {batch_size}")
        
        history = model.fit(
            [X_a_train, X_b_train], y_train,
            validation_data=([X_a_val, X_b_val], y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Salvar modelo
        model_path = self.model_dir / "modelo_assinaturas_manuscritas.h5"
        model.save(str(model_path))
        
        print(f"\n✅ TREINAMENTO CONCLUÍDO!")
        print(f"📁 Modelo salvo em: {model_path}")
        
        # Estatísticas finais
        final_loss = history.history['val_loss'][-1]
        final_acc = history.history['val_accuracy'][-1]
        
        print(f"📊 Performance final:")
        print(f"   Loss de validação: {final_loss:.4f}")
        print(f"   Acurácia de validação: {final_acc:.4f}")
        
        return True

def main():
    print("🤖 TREINAMENTO DE MODELO PARA ASSINATURAS MANUSCRITAS")
    print("=" * 60)
    
    trainer = ModelTrainer()
    
    if trainer.treinar():
        print(f"\n🎉 Sucesso! Próximos passos:")
        print(f"1. Testar: python scripts/avaliar_modelo.py")
        print(f"2. Usar: streamlit run app.py")
    else:
        print(f"\n❌ Falha no treinamento. Verifique os dados.")

if __name__ == "__main__":
    main()
