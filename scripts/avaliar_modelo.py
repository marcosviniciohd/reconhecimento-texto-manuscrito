#!/usr/bin/env python3
"""
Script para avaliar o modelo treinado de verificação de assinaturas manuscritas.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# Adicionar diretório pai ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_preprocessing import preprocess_image
from model import euclidean_distance, contrastive_loss

class ModelEvaluator:
    def __init__(self, model_path="modelos/modelo_assinaturas_manuscritas.h5", 
                 test_data_dir="assinaturas_reais"):
        self.model_path = Path(model_path)
        self.test_data_dir = Path(test_data_dir)
        self.model = None
    
    def carregar_modelo(self):
        """Carrega o modelo treinado."""
        if not self.model_path.exists():
            print(f"❌ Modelo não encontrado em {self.model_path}")
            print("Execute primeiro: python scripts/treinar_modelo.py")
            return False
        
        try:
            # Carregar modelo com funções personalizadas
            self.model = tf.keras.models.load_model(
                str(self.model_path),
                custom_objects={
                    'euclidean_distance': euclidean_distance,
                    'contrastive_loss': contrastive_loss
                }
            )
            print(f"✅ Modelo carregado de {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def euclidean_distance(self, predictions):
        """Calcula distância euclidiana entre vetores de características."""
        return np.sqrt(np.sum(np.square(predictions[0] - predictions[1]), axis=1))
    
    def carregar_dados_teste(self):
        """Carrega imagens de teste das assinaturas reais."""
        if not self.test_data_dir.exists():
            print(f"❌ Dados de teste não encontrados em {self.test_data_dir}")
            print("Adicione assinaturas reais na pasta assinaturas_reais/")
            return None, None
        
        images = []
        labels = []
        
        print("📂 Carregando dados de teste...")
        
        for pessoa_dir in self.test_data_dir.iterdir():
            if not pessoa_dir.is_dir():
                continue
            
            pessoa_name = pessoa_dir.name
            pessoa_images = list(pessoa_dir.glob("*.png")) + list(pessoa_dir.glob("*.jpg"))
            
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
        
        if len(images) == 0:
            print("❌ Nenhuma imagem de teste encontrada")
            return None, None
        
        print(f"✅ Dados de teste carregados: {len(images)} imagens de {len(set(labels))} pessoas")
        return np.array(images), np.array(labels)
    
    def criar_pares_teste(self, images, labels):
        """Cria pares para teste."""
        pairs_a, pairs_b, pair_labels, pair_info = [], [], [], []
        unique_labels = list(set(labels))
        
        print("🔗 Criando pares de teste...")
        
        # Mapear label -> índices
        label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        
        for label_a in unique_labels:
            indices_a = label_to_indices[label_a]
            
            # Pares positivos (mesma pessoa)
            for i in range(len(indices_a)):
                for j in range(i + 1, len(indices_a)):
                    pairs_a.append(images[indices_a[i]])
                    pairs_b.append(images[indices_a[j]])
                    pair_labels.append(0)  # mesma pessoa
                    pair_info.append(f"{label_a} vs {label_a}")
            
            # Pares negativos (pessoas diferentes)
            for label_b in unique_labels:
                if label_a != label_b:
                    indices_b = label_to_indices[label_b]
                    # Apenas alguns pares para não explodir o dataset
                    for i in range(min(3, len(indices_a))):
                        for j in range(min(3, len(indices_b))):
                            pairs_a.append(images[indices_a[i]])
                            pairs_b.append(images[indices_b[j]])
                            pair_labels.append(1)  # pessoas diferentes
                            pair_info.append(f"{label_a} vs {label_b}")
        
        pairs_a = np.array(pairs_a)
        pairs_b = np.array(pairs_b)
        pair_labels = np.array(pair_labels)
        
        print(f"✅ Pares de teste criados: {len(pair_labels)} total")
        print(f"   Pares positivos: {np.sum(pair_labels == 0)}")
        print(f"   Pares negativos: {np.sum(pair_labels == 1)}")
        
        return pairs_a, pairs_b, pair_labels, pair_info
    
    def avaliar_thresholds(self, distancias, labels_reais):
        """Avalia diferentes thresholds para classificação."""
        thresholds = np.arange(0.1, 1.0, 0.05)
        resultados = []
        
        for threshold in thresholds:
            predictions = (distancias <= threshold).astype(int)  # 1 se mesma pessoa
            predictions = 1 - predictions  # Inverter: 0=mesma, 1=diferente
            
            accuracy = np.mean(predictions == labels_reais)
            
            # True positives, false positives, etc.
            tp = np.sum((predictions == 0) & (labels_reais == 0))  # Mesma pessoa correta
            fp = np.sum((predictions == 0) & (labels_reais == 1))  # Falso positivo
            tn = np.sum((predictions == 1) & (labels_reais == 1))  # Diferente correta
            fn = np.sum((predictions == 1) & (labels_reais == 0))  # Falso negativo
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            resultados.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        return resultados
    
    def avaliar(self):
        """Executa avaliação completa do modelo."""
        if not self.carregar_modelo():
            return
        
        # Carregar dados de teste
        images, labels = self.carregar_dados_teste()
        if images is None:
            return
        
        # Criar pares de teste
        pairs_a, pairs_b, pair_labels, pair_info = self.criar_pares_teste(images, labels)
        
        # Fazer predições
        print(f"\n🔮 Executando predições...")
        
        # O modelo siamês retorna a distância diretamente
        distancias = self.model.predict([pairs_a, pairs_b], verbose=0)
        
        # Se retorna array 2D, pegar a primeira coluna
        if distancias.ndim > 1:
            distancias = distancias.flatten()
        
        print(f"✅ Predições concluídas")
        print(f"📊 Estatísticas das distâncias:")
        print(f"   Mínima: {np.min(distancias):.4f}")
        print(f"   Máxima: {np.max(distancias):.4f}")
        print(f"   Média: {np.mean(distancias):.4f}")
        print(f"   Desvio padrão: {np.std(distancias):.4f}")
        
        # Análise de distâncias por categoria
        print(f"\n📈 Análise por categoria:")
        mesma_pessoa = distancias[pair_labels == 0]
        pessoas_diferentes = distancias[pair_labels == 1]
        
        print(f"   Mesma pessoa - Média: {np.mean(mesma_pessoa):.4f}, Desvio: {np.std(mesma_pessoa):.4f}")
        print(f"   Pessoas diferentes - Média: {np.mean(pessoas_diferentes):.4f}, Desvio: {np.std(pessoas_diferentes):.4f}")
        
        # Avaliar diferentes thresholds
        print(f"\n🎯 Avaliando thresholds...")
        resultados = self.avaliar_thresholds(distancias, pair_labels)
        
        # Encontrar melhor threshold
        melhor = max(resultados, key=lambda x: x['f1'])
        
        print(f"\n🏆 MELHOR THRESHOLD: {melhor['threshold']:.3f}")
        print(f"   Acurácia: {melhor['accuracy']:.3f}")
        print(f"   Precisão: {melhor['precision']:.3f}")
        print(f"   Recall: {melhor['recall']:.3f}")
        print(f"   F1-Score: {melhor['f1']:.3f}")
        
        # Top 5 thresholds
        top_5 = sorted(resultados, key=lambda x: x['f1'], reverse=True)[:5]
        print(f"\n🏅 TOP 5 THRESHOLDS:")
        for i, res in enumerate(top_5, 1):
            print(f"   {i}. Threshold: {res['threshold']:.3f} | F1: {res['f1']:.3f} | Acc: {res['accuracy']:.3f}")
        
        # Classificação com melhor threshold
        threshold_otimo = melhor['threshold']
        predictions_final = (distancias <= threshold_otimo).astype(int)
        predictions_final = 1 - predictions_final  # Inverter
        
        print(f"\n📋 RELATÓRIO DETALHADO (Threshold: {threshold_otimo:.3f}):")
        print(classification_report(
            pair_labels, predictions_final,
            target_names=['Mesma Pessoa', 'Pessoas Diferentes']
        ))
        
        # Salvar resultados
        resultados_dir = Path("resultados_avaliacao")
        resultados_dir.mkdir(exist_ok=True)
        
        # Salvar threshold ótimo
        with open(resultados_dir / "threshold_otimo.txt", "w") as f:
            f.write(f"{threshold_otimo:.4f}\n")
        
        print(f"💾 Threshold ótimo salvo em: resultados_avaliacao/threshold_otimo.txt")
        print(f"💡 Use este valor no app.py para melhor performance!")
        
        return threshold_otimo

def main():
    print("🔍 AVALIAÇÃO DO MODELO DE ASSINATURAS MANUSCRITAS")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    threshold_otimo = evaluator.avaliar()
    
    if threshold_otimo:
        print(f"\n✅ Avaliação concluída!")
        print(f"🎯 Threshold recomendado: {threshold_otimo:.4f}")

if __name__ == "__main__":
    main()
