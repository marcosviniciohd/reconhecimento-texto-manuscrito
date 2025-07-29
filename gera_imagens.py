import numpy as np
import cv2
import os
import random

autores = ['autor1', 'autor2', 'autor3']
nomes = ['Joao', 'Maria', 'Carlos']

for autor, nome in zip(autores, nomes):
    dir_path = f'data/{autor}'
    os.makedirs(dir_path, exist_ok=True)
    
    # Gerar 10 assinaturas por autor com variações
    for i in range(1, 11):
        img = np.ones((155, 220), dtype=np.uint8) * 255
        
        # Variações aleatórias
        x = random.randint(5, 20)
        y = random.randint(70, 90)
        scale = random.uniform(1.5, 2.5)
        thickness = random.randint(2, 5)
        
        fonte = random.choice([cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 
                              cv2.FONT_HERSHEY_SIMPLEX,
                              cv2.FONT_HERSHEY_COMPLEX])
        
        cv2.putText(img, nome, (x, y), fonte, scale, (0), thickness, cv2.LINE_AA)
        cv2.imwrite(f'{dir_path}/assinatura{i}.png', img)