import numpy as np
import cv2
import os

autores = ['autor1', 'autor2', 'autor3']
nomes = ['Joao', 'Maria', 'Carlos']

for autor, nome in zip(autores, nomes):
    dir_path = f'data/{autor}'
    os.makedirs(dir_path, exist_ok=True)
    for i in range(1, 3):
        img = np.ones((155, 220), dtype=np.uint8) * 255  # (altura, largura)
        x = 10 + 10*i
        y = 80 + 10*i
        fonte = cv2.FONT_HERSHEY_SCRIPT_COMPLEX if i % 2 else cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, nome, (x, y), fonte, 2, (0), 4, cv2.LINE_AA)
        cv2.imwrite(f'{dir_path}/assinatura{i}.png', img)