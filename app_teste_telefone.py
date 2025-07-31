#!/usr/bin/env python3
"""
Sistema de teste para assinaturas capturadas por telefone vs registradas.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import sys
import cv2
from PIL import Image

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import binarize_image, resize_image, normalize_image

st.set_page_config(
    page_title="📱 Teste Assinaturas por Telefone",
    page_icon="📱",
    layout="wide"
)

def preprocess_phone_image(uploaded_file, enhance_quality=True):
    """Preprocessa imagem capturada por telefone com melhorias."""
    try:
        # Converter arquivo Streamlit para PIL Image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Converter para escala de cinza se necessário
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Melhorias específicas para fotos de telefone
        if enhance_quality:
            # Redução de ruído
            img_array = cv2.medianBlur(img_array, 3)
            
            # Melhorar contraste
            img_array = cv2.equalizeHist(img_array)
            
            # Detecção automática de threshold para binarização
            threshold_value = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        else:
            threshold_value = 127
        
        # Aplicar processamento padrão
        img_processed = binarize_image(img_array, threshold=threshold_value)
        img_processed = resize_image(img_processed)
        img_processed = normalize_image(img_processed)
        
        return img_processed, threshold_value
    except Exception as e:
        raise Exception(f"Erro no preprocessamento: {str(e)}")

class PhoneSignatureVerifier:
    def __init__(self):
        self.model = None
        self.threshold = 0.10  # Threshold otimizado
        self.registered_signatures = {}
        
    def carregar_modelo(self):
        """Carrega o modelo treinado."""
        model_path = Path("modelos/modelo_assinaturas_manuscritas.h5")
        
        if not model_path.exists():
            return False
        
        try:
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
    
    def carregar_assinaturas_registradas(self):
        """Carrega assinaturas já registradas no sistema."""
        signatures_dir = Path("assinaturas_reais")
        
        if not signatures_dir.exists():
            return {}
        
        registered = {}
        
        for pessoa_dir in signatures_dir.iterdir():
            if not pessoa_dir.is_dir():
                continue
            
            pessoa_name = pessoa_dir.name
            pessoa_images = list(pessoa_dir.glob("*.png")) + list(pessoa_dir.glob("*.jpg"))
            
            if len(pessoa_images) > 0:
                registered[pessoa_name] = pessoa_images
        
        return registered
    
    def verificar_contra_registradas(self, phone_image, pessoa_selecionada=None):
        """Verifica assinatura do telefone contra registradas."""
        if self.model is None:
            return None, "Modelo não carregado"
        
        signatures = self.carregar_assinaturas_registradas()
        
        if not signatures:
            return None, "Nenhuma assinatura registrada encontrada"
        
        try:
            # Preprocessar imagem do telefone
            phone_processed, phone_threshold = preprocess_phone_image(phone_image)
            
            if phone_processed.ndim == 2:
                phone_processed = np.expand_dims(phone_processed, axis=-1)
            phone_processed = np.expand_dims(phone_processed, axis=0)
            
            resultados = []
            
            # Testar contra pessoas específicas ou todas
            pessoas_testar = [pessoa_selecionada] if pessoa_selecionada else signatures.keys()
            
            for pessoa in pessoas_testar:
                if pessoa not in signatures:
                    continue
                
                pessoa_resultados = []
                
                # Testar contra todas as assinaturas registradas da pessoa
                for signature_path in signatures[pessoa]:
                    try:
                        # Preprocessar assinatura registrada
                        from data_preprocessing import preprocess_image
                        registered_img = preprocess_image(str(signature_path))
                        
                        if registered_img.ndim == 2:
                            registered_img = np.expand_dims(registered_img, axis=-1)
                        registered_img = np.expand_dims(registered_img, axis=0)
                        
                        # Calcular distância
                        distance = self.model.predict([phone_processed, registered_img], verbose=0)[0]
                        
                        if hasattr(distance, '__len__'):
                            distance = distance[0]
                        
                        pessoa_resultados.append({
                            'arquivo': signature_path.name,
                            'distancia': distance,
                            'mesma_pessoa': distance <= self.threshold
                        })
                        
                    except Exception as e:
                        st.warning(f"Erro ao processar {signature_path}: {e}")
                
                if pessoa_resultados:
                    # Estatísticas da pessoa
                    distancias = [r['distancia'] for r in pessoa_resultados]
                    media_dist = np.mean(distancias)
                    min_dist = np.min(distancias)
                    max_dist = np.max(distancias)
                    
                    # Decisão: maioria ou distância mínima
                    matches = sum(1 for r in pessoa_resultados if r['mesma_pessoa'])
                    total_tests = len(pessoa_resultados)
                    
                    resultados.append({
                        'pessoa': pessoa,
                        'testes_individuais': pessoa_resultados,
                        'media_distancia': media_dist,
                        'min_distancia': min_dist,
                        'max_distancia': max_dist,
                        'matches': matches,
                        'total_tests': total_tests,
                        'percentual_match': (matches / total_tests) * 100,
                        'melhor_match': min_dist <= self.threshold,
                        'phone_threshold': phone_threshold
                    })
            
            return resultados, None
            
        except Exception as e:
            return None, f"Erro na verificação: {str(e)}"

def main():
    st.title("📱 Verificação de Assinaturas por Telefone")
    st.markdown("---")
    
    st.markdown("""
    ### 📝 Cenário de Teste:
    - **Assinaturas Registradas**: Já cadastradas no sistema (papel, scanner)
    - **Assinatura Teste**: Capturada por telefone (câmera, diferentes condições)
    
    ### 🎯 Objetivo:
    Verificar se a assinatura fotografada corresponde a alguma pessoa registrada.
    """)
    
    # Inicializar verificador
    if 'verifier' not in st.session_state:
        st.session_state.verifier = PhoneSignatureVerifier()
    
    verifier = st.session_state.verifier
    
    # Verificar se modelo está carregado
    if verifier.model is None:
        with st.spinner("Carregando modelo..."):
            if verifier.carregar_modelo():
                st.success("✅ Modelo carregado!")
            else:
                st.error("❌ Modelo não encontrado. Execute primeiro: `python scripts/treinar_modelo.py`")
                st.stop()
    
    # Carregar assinaturas registradas
    signatures = verifier.carregar_assinaturas_registradas()
    
    if not signatures:
        st.error("❌ Nenhuma assinatura registrada encontrada em `assinaturas_reais/`")
        st.stop()
    
    # Mostrar pessoas registradas
    st.sidebar.markdown("### 👥 Pessoas Registradas:")
    for pessoa, files in signatures.items():
        st.sidebar.write(f"**{pessoa}**: {len(files)} assinaturas")
    
    # Seleção de pessoa (opcional)
    st.markdown("### 🎯 Opções de Teste:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_mode = st.radio(
            "Modo de teste:",
            ["🔍 Testar contra TODAS as pessoas", "👤 Testar contra pessoa específica"]
        )
    
    with col2:
        pessoa_especifica = None
        if "específica" in test_mode:
            pessoa_especifica = st.selectbox(
                "Selecione a pessoa:",
                list(signatures.keys())
            )
    
    # Upload da assinatura do telefone
    st.markdown("### 📱 Upload da Assinatura Capturada:")
    
    phone_image = st.file_uploader(
        "Envie a foto da assinatura (capturada por telefone)",
        type=['png', 'jpg', 'jpeg'],
        help="Foto tirada com câmera do celular da assinatura em papel"
    )
    
    if phone_image:
        # Opções de processamento
        st.markdown("### ⚙️ Opções de Processamento:")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            enhance_quality = st.checkbox(
                "📈 Melhorar qualidade da imagem",
                value=True,
                help="Aplica filtros para melhorar fotos de telefone"
            )
        
        with col_opt2:
            show_processing = st.checkbox(
                "🔬 Mostrar etapas do processamento",
                value=False
            )
        
        # Mostrar preview da imagem
        st.image(phone_image, caption="📱 Assinatura Capturada", width=300)
        
        # Botão de verificação
        if st.button("🔍 Verificar Assinatura", use_container_width=True):
            with st.spinner("Analisando assinatura..."):
                
                # Mostrar processamento se solicitado
                if show_processing:
                    st.markdown("#### 🔬 Processamento da Imagem:")
                    
                    processed_img, threshold_used = preprocess_phone_image(phone_image, enhance_quality)
                    
                    col_proc1, col_proc2 = st.columns(2)
                    with col_proc1:
                        st.image(phone_image, caption="Original", width=200)
                    with col_proc2:
                        # Converter para exibição
                        display_img = (processed_img * 255).astype(np.uint8)
                        st.image(display_img, caption=f"Processada (threshold: {threshold_used})", width=200)
                
                # Executar verificação
                resultados, erro = verifier.verificar_contra_registradas(
                    phone_image, pessoa_especifica
                )
                
                if erro:
                    st.error(f"❌ {erro}")
                else:
                    st.markdown("---")
                    st.markdown("## 📊 Resultados da Verificação:")
                    
                    # Ordenar por melhor match
                    resultados.sort(key=lambda x: x['min_distancia'])
                    
                    for i, resultado in enumerate(resultados):
                        pessoa = resultado['pessoa']
                        
                        # Determinar status
                        if resultado['melhor_match']:
                            status = "✅ MATCH ENCONTRADO"
                            status_color = "success"
                        elif resultado['percentual_match'] >= 50:
                            status = "⚠️ POSSÍVEL MATCH"
                            status_color = "warning"
                        else:
                            status = "❌ NÃO CORRESPONDE"
                            status_color = "error"
                        
                        # Exibir resultado
                        with st.container():
                            if status_color == "success":
                                st.success(f"**{pessoa}**: {status}")
                            elif status_color == "warning":
                                st.warning(f"**{pessoa}**: {status}")
                            else:
                                st.error(f"**{pessoa}**: {status}")
                            
                            # Métricas detalhadas
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            
                            with col_m1:
                                st.metric("Melhor Distância", f"{resultado['min_distancia']:.4f}")
                            with col_m2:
                                st.metric("Distância Média", f"{resultado['media_distancia']:.4f}")
                            with col_m3:
                                st.metric("Matches", f"{resultado['matches']}/{resultado['total_tests']}")
                            with col_m4:
                                st.metric("% Compatibilidade", f"{resultado['percentual_match']:.1f}%")
                            
                            # Testes individuais
                            with st.expander(f"🔍 Detalhes dos testes - {pessoa}"):
                                for teste in resultado['testes_individuais']:
                                    col_t1, col_t2, col_t3 = st.columns([2, 1, 1])
                                    
                                    with col_t1:
                                        st.write(f"📄 {teste['arquivo']}")
                                    with col_t2:
                                        st.write(f"📏 {teste['distancia']:.4f}")
                                    with col_t3:
                                        st.write("✅" if teste['mesma_pessoa'] else "❌")
                    
                    # Interpretação geral
                    st.markdown("### 💡 Interpretação:")
                    
                    best_result = resultados[0] if resultados else None
                    
                    if best_result and best_result['melhor_match']:
                        st.info(f"🎯 **Resultado Mais Provável**: A assinatura corresponde a **{best_result['pessoa']}**")
                    elif best_result and best_result['percentual_match'] >= 30:
                        st.warning(f"❓ **Resultado Inconclusivo**: Possível correspondência com **{best_result['pessoa']}**, mas verificação manual recomendada")
                    else:
                        st.error("🚫 **Assinatura não reconhecida**: Não corresponde a nenhuma pessoa registrada no sistema")
                    
                    # Recomendações
                    st.markdown("### 📋 Recomendações:")
                    
                    if best_result:
                        if best_result['min_distancia'] <= 0.05:
                            st.success("• **Alta confiança** - Resultado muito confiável")
                        elif best_result['min_distancia'] <= 0.15:
                            st.warning("• **Confiança moderada** - Considere verificação adicional")
                        else:
                            st.error("• **Baixa confiança** - Recomenda-se análise manual")
                    
                    st.info("• Para melhorar a precisão: tire fotos com boa iluminação, fundo claro e sem sombras")
                    st.info("• Mantenha a câmera paralela ao papel para evitar distorções")

if __name__ == "__main__":
    main()
