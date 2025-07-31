#!/usr/bin/env python3
"""
Script para preparar dataset de assinaturas manuscritas.
Aplica data augmentation e organiza os dados para treinamento.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Adicionar diretório pai ao path para importar data_preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_preprocessing import preprocess_image

class DatasetPreparator:
    def __init__(self, input_dir="assinaturas_reais", output_dir="dataset_processado"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def aplicar_augmentation(self, img_path, output_folder, base_name):
        """Aplica data augmentation em uma imagem."""
        img_orig = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_orig is None:
            print(f"❌ Erro ao carregar: {img_path}")
            return 0
        
        count = 0
        
        # 1. Original
        img_proc = preprocess_image(str(img_path))
        cv2.imwrite(str(output_folder / f"{base_name}_original.png"), 
                   (img_proc * 255).astype(np.uint8))
        count += 1
        
        # 2. Rotações
        for angle in [-3, -1, 1, 3, 5]:
            center = (img_orig.shape[1]//2, img_orig.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_orig, rotation_matrix, 
                                   (img_orig.shape[1], img_orig.shape[0]), 
                                   borderValue=255)
            
            temp_path = output_folder / f"temp_rot_{angle}.png"
            cv2.imwrite(str(temp_path), rotated)
            
            img_proc = preprocess_image(str(temp_path))
            cv2.imwrite(str(output_folder / f"{base_name}_rot{angle:+d}.png"), 
                       (img_proc * 255).astype(np.uint8))
            temp_path.unlink()
            count += 1
        
        # 3. Escalas
        for scale in [0.95, 0.98, 1.02, 1.05]:
            scaled_h = int(img_orig.shape[0] * scale)
            scaled_w = int(img_orig.shape[1] * scale)
            scaled = cv2.resize(img_orig, (scaled_w, scaled_h))
            
            if scale < 1.0:
                pad_h = (img_orig.shape[0] - scaled_h) // 2
                pad_w = (img_orig.shape[1] - scaled_w) // 2
                scaled_final = cv2.copyMakeBorder(scaled, pad_h, pad_h, pad_w, pad_w, 
                                                cv2.BORDER_CONSTANT, value=255)
            else:
                start_h = (scaled_h - img_orig.shape[0]) // 2
                start_w = (scaled_w - img_orig.shape[1]) // 2
                scaled_final = scaled[start_h:start_h + img_orig.shape[0], 
                                    start_w:start_w + img_orig.shape[1]]
            
            temp_path = output_folder / f"temp_scale_{scale}.png"
            cv2.imwrite(str(temp_path), scaled_final)
            
            img_proc = preprocess_image(str(temp_path))
            cv2.imwrite(str(output_folder / f"{base_name}_scale{scale:.2f}.png"), 
                       (img_proc * 255).astype(np.uint8))
            temp_path.unlink()
            count += 1
        
        # 4. Translações
        for dx, dy in [(-3, 0), (3, 0), (0, -2), (0, 2), (-2, -1), (2, 1)]:
            translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            translated = cv2.warpAffine(img_orig, translation_matrix, 
                                      (img_orig.shape[1], img_orig.shape[0]),
                                      borderValue=255)
            
            temp_path = output_folder / f"temp_trans_{dx}_{dy}.png"
            cv2.imwrite(str(temp_path), translated)
            
            img_proc = preprocess_image(str(temp_path))
            cv2.imwrite(str(output_folder / f"{base_name}_trans{dx:+d}{dy:+d}.png"), 
                       (img_proc * 255).astype(np.uint8))
            temp_path.unlink()
            count += 1
        
        # 5. Ruído leve
        for noise_level in [0.01, 0.03]:
            noisy = img_orig.copy().astype(np.float32)
            noise = np.random.normal(0, noise_level * 255, img_orig.shape)
            noisy = noisy + noise
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            
            temp_path = output_folder / f"temp_noise_{noise_level}.png"
            cv2.imwrite(str(temp_path), noisy)
            
            img_proc = preprocess_image(str(temp_path))
            cv2.imwrite(str(output_folder / f"{base_name}_noise{noise_level:.2f}.png"), 
                       (img_proc * 255).astype(np.uint8))
            temp_path.unlink()
            count += 1
        
        return count
    
    def processar(self):
        """Processa todas as assinaturas encontradas."""
        if not self.input_dir.exists():
            print(f"❌ Pasta {self.input_dir} não encontrada!")
            print("📋 Como usar:")
            print("1. Crie a pasta 'assinaturas_reais/'")
            print("2. Crie subpastas para cada pessoa: pessoa1/, pessoa2/, etc.")
            print("3. Coloque fotos das assinaturas em cada subpasta")
            print("4. Execute este script novamente")
            return False
        
        total_imagens = 0
        total_pessoas = 0
        
        print("🔄 PREPARANDO DATASET DE ASSINATURAS MANUSCRITAS")
        print("=" * 50)
        
        for pessoa_dir in self.input_dir.iterdir():
            if not pessoa_dir.is_dir():
                continue
            
            pessoa_name = pessoa_dir.name
            output_pessoa = self.output_dir / pessoa_name
            output_pessoa.mkdir(exist_ok=True)
            
            # Buscar imagens
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(pessoa_dir.glob(ext))
                image_files.extend(pessoa_dir.glob(ext.upper()))
            
            if not image_files:
                print(f"⚠️ Nenhuma imagem encontrada em {pessoa_dir}")
                continue
            
            print(f"\n👤 Processando: {pessoa_name}")
            print(f"   Imagens originais: {len(image_files)}")
            
            pessoa_count = 0
            for i, img_file in enumerate(image_files, 1):
                base_name = f"{pessoa_name}_img{i:02d}"
                count = self.aplicar_augmentation(img_file, output_pessoa, base_name)
                pessoa_count += count
                print(f"   ✅ {img_file.name}: {count} variações")
            
            print(f"   📊 Total para {pessoa_name}: {pessoa_count} imagens")
            total_imagens += pessoa_count
            total_pessoas += 1
        
        print(f"\n🎉 DATASET PREPARADO!")
        print(f"📊 Estatísticas finais:")
        print(f"   Pessoas: {total_pessoas}")
        print(f"   Total de imagens: {total_imagens}")
        print(f"   Pasta de saída: {self.output_dir}")
        
        if total_pessoas < 2:
            print(f"\n⚠️ ATENÇÃO: Você precisa de pelo menos 2 pessoas para treinar!")
            print(f"   Adicione mais pastas em {self.input_dir}")
            return False
        
        return True

def main():
    print("📝 PREPARADOR DE DATASET PARA ASSINATURAS MANUSCRITAS")
    print("=" * 55)
    
    preparador = DatasetPreparator()
    
    if preparador.processar():
        print(f"\n✅ Pronto! Agora execute: python scripts/treinar_modelo.py")
    else:
        print(f"\n❌ Falha na preparação. Verifique as instruções acima.")

if __name__ == "__main__":
    main()
