#!/usr/bin/env python3
"""
Aplicação Streamlit para verificação de assinaturas manuscritas.
Interface limpa e focada em assinaturas reais.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import sys
import cv2
from PIL import Image

from data_preprocessing import binarize_image, resize_image, normalize_image

# Configuração da página
st.set_page_config(
    page_title="Verificação de Assinaturas Manuscritas",
    page_icon="✍️",
    layout="wide"
)

def preprocess_streamlit_image(uploaded_file):
    """Preprocessa imagem carregada via Streamlit."""
    try:
        # Converter arquivo Streamlit para PIL Image
        image = Image.open(uploaded_file)
        
        # Converter para numpy array
        img_array = np.array(image)
        
        # Se a imagem tem 3 canais (RGB), converter para escala de cinza
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Aplicar processamento igual ao data_preprocessing.py
        img_processed = binarize_image(img_array)
        img_processed = resize_image(img_processed)
        img_processed = normalize_image(img_processed)
        
        return img_processed
    except Exception as e:
        raise Exception(f"Erro no preprocessamento: {str(e)}")

class SignatureVerifier:
    def __init__(self):
        self.model = None
        self.threshold = 0.10  # Valor otimizado pela avaliação
        
    def carregar_modelo(self):
        """Carrega o modelo treinado."""
        model_path = Path("modelos/modelo_assinaturas_manuscritas.h5")
        
        if not model_path.exists():
            return False
        
        try:
            # Importar funções personalizadas
            from model import euclidean_distance, contrastive_loss
            
            self.model = tf.keras.models.load_model(
                str(model_path),
                custom_objects={
                    'euclidean_distance': euclidean_distance,
                    'contrastive_loss': contrastive_loss
                }
            )
            return True
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def carregar_threshold_otimo(self):
        """Carrega threshold ótimo se disponível."""
        threshold_path = Path("resultados_avaliacao/threshold_otimo.txt")
        
        if threshold_path.exists():
            try:
                with open(threshold_path, 'r') as f:
                    self.threshold = float(f.read().strip())
                return True
            except:
                pass
        return False
    
    def euclidean_distance(self, predictions):
        """Calcula distância euclidiana."""
        return np.sqrt(np.sum(np.square(predictions[0] - predictions[1]), axis=1))
    
    def verificar_assinaturas(self, img1, img2):
        """Verifica se duas assinaturas são da mesma pessoa."""
        if self.model is None:
            return None, "Modelo não carregado"
        
        try:
            # Preprocessar imagens do Streamlit
            proc_img1 = preprocess_streamlit_image(img1)
            proc_img2 = preprocess_streamlit_image(img2)
            
            if proc_img1.ndim == 2:
                proc_img1 = np.expand_dims(proc_img1, axis=-1)
            if proc_img2.ndim == 2:
                proc_img2 = np.expand_dims(proc_img2, axis=-1)
            
            # Adicionar dimensão batch
            proc_img1 = np.expand_dims(proc_img1, axis=0)
            proc_img2 = np.expand_dims(proc_img2, axis=0)
            
            # Calcular distância usando o modelo siamês
            distance = self.model.predict([proc_img1, proc_img2], verbose=0)[0]
            
            # Se retorna array, pegar primeiro valor
            if hasattr(distance, '__len__'):
                distance = distance[0]
            
            # Classificar
            mesma_pessoa = distance <= self.threshold
            confianca = 1 - (distance / (self.threshold * 2))  # Normalizada
            confianca = max(0, min(1, confianca))
            
            return {
                'mesma_pessoa': mesma_pessoa,
                'distancia': distance,
                'confianca': confianca,
                'threshold_usado': self.threshold
            }, None
            
        except Exception as e:
            return None, f"Erro na verificação: {str(e)}"

def main():
    # Título e descrição
    st.title("✍️ Verificação de Assinaturas Manuscritas")
    st.markdown("---")
    
    st.markdown("""
    ### 📝 Como usar:
    1. **Carregue duas imagens** de assinaturas manuscritas
    2. **Clique em 'Verificar'** para comparar
    3. **Veja o resultado** da análise
    
    ⚠️ **Importante**: Use apenas assinaturas manuscritas reais para melhor precisão.
    """)
    
    # Inicializar verificador
    if 'verifier' not in st.session_state:
        st.session_state.verifier = SignatureVerifier()
    
    verifier = st.session_state.verifier
    
    # Verificar se modelo está carregado
    if verifier.model is None:
        with st.spinner("Carregando modelo..."):
            if verifier.carregar_modelo():
                verifier.carregar_threshold_otimo()
                st.success("✅ Modelo carregado com sucesso!")
            else:
                st.error("❌ **Modelo não encontrado!**")
                st.info("Execute primeiro: `python scripts/treinar_modelo.py`")
                st.stop()
    
    # Interface de upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Assinatura 1")
        uploaded_file1 = st.file_uploader(
            "Escolha a primeira assinatura",
            type=['png', 'jpg', 'jpeg'],
            key="file1"
        )
        
        if uploaded_file1:
            st.image(uploaded_file1, caption="Assinatura 1", use_container_width=True)
    
    with col2:
        st.subheader("📷 Assinatura 2")
        uploaded_file2 = st.file_uploader(
            "Escolha a segunda assinatura",
            type=['png', 'jpg', 'jpeg'],
            key="file2"
        )
        
        if uploaded_file2:
            st.image(uploaded_file2, caption="Assinatura 2", use_container_width=True)
    
    # Botão de verificação
    if uploaded_file1 and uploaded_file2:
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔍 Verificar Assinaturas", use_container_width=True):
                with st.spinner("Analisando assinaturas..."):
                    resultado, erro = verifier.verificar_assinaturas(
                        uploaded_file1, uploaded_file2
                    )
                
                st.markdown("---")
                
                if erro:
                    st.error(f"❌ {erro}")
                else:
                    # Exibir resultado
                    if resultado['mesma_pessoa']:
                        st.success("✅ **MESMA PESSOA**")
                        st.balloons()
                    else:
                        st.error("❌ **PESSOAS DIFERENTES**")
                    
                    # Métricas detalhadas
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric(
                            "Distância Calculada",
                            f"{resultado['distancia']:.4f}"
                        )
                    
                    with col_m2:
                        st.metric(
                            "Threshold Usado",
                            f"{resultado['threshold_usado']:.4f}"
                        )
                    
                    with col_m3:
                        confianca_pct = resultado['confianca'] * 100
                        st.metric(
                            "Confiança",
                            f"{confianca_pct:.1f}%"
                        )
                    
                    # Interpretação
                    st.markdown("### 📊 Interpretação:")
                    
                    if resultado['confianca'] >= 0.8:
                        st.info("🎯 **Alta confiança** no resultado")
                    elif resultado['confianca'] >= 0.6:
                        st.warning("⚠️ **Confiança média** - verificar manualmente")
                    else:
                        st.warning("❓ **Baixa confiança** - resultado incerto")
                    
                    # Explicação técnica
                    with st.expander("🔬 Detalhes Técnicos"):
                        st.write(f"**Distância euclidiana:** {resultado['distancia']:.6f}")
                        st.write(f"**Threshold de decisão:** {resultado['threshold_usado']:.6f}")
                        st.write(f"**Lógica:** {'Distância ≤ Threshold' if resultado['mesma_pessoa'] else 'Distância > Threshold'}")
                        
                        if resultado['distancia'] <= resultado['threshold_usado']:
                            st.write("✅ Assinaturas são **similares o suficiente** para serem da mesma pessoa")
                        else:
                            st.write("❌ Assinaturas são **muito diferentes** para serem da mesma pessoa")
    
    # Rodapé com informações
    st.markdown("---")
    
    with st.expander("ℹ️ Informações do Sistema"):
        st.write("**Modelo:** Rede Neural Siamesa")
        st.write("**Função de Perda:** Contrastive Loss")
        st.write("**Focado em:** Assinaturas manuscritas reais")
        
        if verifier.carregar_threshold_otimo():
            st.write(f"**Threshold otimizado:** {verifier.threshold:.4f} (calibrado via avaliação)")
        else:
            st.write(f"**Threshold padrão:** {verifier.threshold:.4f}")
        
        st.write("**Para melhor performance:** Execute `python scripts/avaliar_modelo.py` para calibrar threshold")

if __name__ == "__main__":
    main()
