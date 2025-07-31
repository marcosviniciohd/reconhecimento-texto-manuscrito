#!/usr/bin/env python3
"""
Script para analisar problemas nos dados de treinamento.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Adicionar diretório pai ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_preprocessing import preprocess_image

def analisar_dataset():
    """Analisa a estrutura do dataset para identificar problemas."""
    
    print("🔍 ANÁLISE DO DATASET")
    print("=" * 50)
    
    data_dir = Path("assinaturas_reais")
    
    if not data_dir.exists():
        print("❌ Pasta assinaturas_reais não encontrada")
        return
    
    pessoas = []
    total_imagens = 0
    problemas = []
    
    for pessoa_dir in data_dir.iterdir():
        if not pessoa_dir.is_dir():
            continue
        
        pessoa_name = pessoa_dir.name
        pessoa_images = list(pessoa_dir.glob("*.png")) + list(pessoa_dir.glob("*.jpg"))
        num_images = len(pessoa_images)
        
        pessoas.append({
            'nome': pessoa_name,
            'num_imagens': num_images,
            'images': pessoa_images
        })
        
        total_imagens += num_images
        
        print(f"👤 {pessoa_name}: {num_images} imagens")
        
        # Verificar se há imagens suficientes para pares positivos
        if num_images < 2:
            problemas.append(f"❌ {pessoa_name}: Apenas {num_images} imagem(ns) - Impossível criar pares positivos")
        elif num_images < 5:
            problemas.append(f"⚠️ {pessoa_name}: Apenas {num_images} imagens - Poucos pares positivos")
    
    print(f"\n📊 Resumo:")
    print(f"   Total de pessoas: {len(pessoas)}")
    print(f"   Total de imagens: {total_imagens}")
    
    # Calcular pares que serão criados
    pares_positivos = 0
    pares_negativos = 0
    
    for pessoa in pessoas:
        n = pessoa['num_imagens']
        # Pares positivos: C(n,2) = n*(n-1)/2
        pares_pos_pessoa = n * (n - 1) // 2
        pares_positivos += pares_pos_pessoa
        
        print(f"   {pessoa['nome']}: {pares_pos_pessoa} pares positivos")
        
        # Pares negativos: para cada pessoa, 3x3 com cada outra pessoa
        for outra_pessoa in pessoas:
            if pessoa['nome'] != outra_pessoa['nome']:
                pares_neg = min(3, n) * min(3, outra_pessoa['num_imagens'])
                pares_negativos += pares_neg
    
    print(f"\n🔗 Pares de treinamento:")
    print(f"   Pares positivos (mesma pessoa): {pares_positivos}")
    print(f"   Pares negativos (pessoas diferentes): {pares_negativos}")
    print(f"   Total de pares: {pares_positivos + pares_negativos}")
    
    # Verificar balanceamento
    if pares_positivos == 0:
        problemas.append("❌ CRÍTICO: Nenhum par positivo será criado!")
    elif pares_negativos / pares_positivos > 10:
        problemas.append(f"⚠️ Dataset desbalanceado: {pares_negativos/pares_positivos:.1f}x mais pares negativos")
    
    # Exibir problemas
    if problemas:
        print(f"\n⚠️ PROBLEMAS DETECTADOS:")
        for problema in problemas:
            print(f"   {problema}")
    else:
        print(f"\n✅ Dataset parece estar bem estruturado!")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES:")
    
    if pares_positivos < 20:
        print(f"   📈 Adicione mais assinaturas por pessoa (ideal: 5-10 por pessoa)")
    
    if len(pessoas) < 3:
        print(f"   👥 Adicione mais pessoas diferentes (ideal: 3-5 pessoas)")
    
    print(f"   📁 Estrutura ideal:")
    print(f"      assinaturas_reais/")
    print(f"      ├── pessoa_A/")
    print(f"      │   ├── assinatura1.png")
    print(f"      │   ├── assinatura2.png") 
    print(f"      │   └── assinatura3.png")
    print(f"      ├── pessoa_B/")
    print(f"      │   ├── assinatura1.png")
    print(f"      │   └── assinatura2.png")
    print(f"      └── pessoa_C/...")
    
    return pessoas, problemas

def verificar_qualidade_dados():
    """Verifica se todas as pessoas têm realmente assinaturas da mesma pessoa."""
    
    print(f"\n🔬 VERIFICAÇÃO DE QUALIDADE DOS DADOS")
    print("=" * 50)
    
    print(f"⚠️  IMPORTANTE: Este script não pode verificar automaticamente se")
    print(f"   as assinaturas dentro de cada pasta são realmente da mesma pessoa.")
    print(f"   Você precisa verificar manualmente:")
    
    data_dir = Path("assinaturas_reais")
    
    for pessoa_dir in data_dir.iterdir():
        if not pessoa_dir.is_dir():
            continue
        
        pessoa_name = pessoa_dir.name
        pessoa_images = list(pessoa_dir.glob("*.png")) + list(pessoa_dir.glob("*.jpg"))
        
        print(f"\n👤 {pessoa_name}:")
        for img_path in pessoa_images:
            print(f"   📄 {img_path.name}")
        
        print(f"   ❓ Todas essas assinaturas são da MESMA pessoa?")
    
    print(f"\n💡 Se NÃO:")
    print(f"   1. Reorganize as pastas por pessoa real")
    print(f"   2. Retreine o modelo: python scripts/treinar_modelo.py")
    print(f"   3. Reavalie: python scripts/avaliar_modelo.py")

def main():
    pessoas, problemas = analisar_dataset()
    verificar_qualidade_dados()
    
    if problemas:
        print(f"\n🚨 AÇÃO NECESSÁRIA:")
        print(f"   Corrija os problemas identificados antes de treinar novamente!")

if __name__ == "__main__":
    main()
