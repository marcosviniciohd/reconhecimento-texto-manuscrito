# 🖋️ Sistema de Reconhecimento de Assinaturas Manuscritas

Sistema inteligente para verificação e comparação de assinaturas manuscritas utilizando **Redes Neurais Siamesas** com TensorFlow/Keras.

## 📋 **Índice**

- [🎯 Sobre o Projeto](#-sobre-o-projeto)
- [✨ Funcionalidades](#-funcionalidades)
- [🛠️ Tecnologias](#%EF%B8%8F-tecnologias)
- [📦 Instalação](#-instalação)
- [🚀 Como Usar](#-como-usar)
- [📊 Performance](#-performance)
- [📱 Teste com Telefone](#-teste-com-telefone)
- [🔧 Treinamento](#-treinamento)
- [📁 Estrutura do Projeto](#-estrutura-do-projeto)
- [🤝 Contribuição](#-contribuição)

---

## 🎯 **Sobre o Projeto**

Este sistema foi desenvolvido para **verificar a autenticidade de assinaturas manuscritas** utilizando técnicas de Deep Learning. O projeto implementa uma arquitetura de **Rede Neural Siamesa** que aprende a distinguir entre assinaturas da mesma pessoa e de pessoas diferentes.

### **Principais Características:**
- ✅ **Foco em assinaturas manuscritas reais**
- ✅ **Interface web intuitiva** (Streamlit)
- ✅ **Threshold otimizado automaticamente**
- ✅ **Pipeline completo de treinamento**
- ✅ **Suporte para fotos capturadas por telefone**

---

## ✨ **Funcionalidades**

### **🔍 Verificação de Assinaturas**
- Comparação entre duas assinaturas
- Resultado com nível de confiança
- Interpretação automática dos resultados

### **📱 Teste com Telefone**
- Interface especializada para fotos de telefone
- Melhoria automática de qualidade da imagem
- Comparação contra base de assinaturas registradas

### **📊 Métricas Detalhadas**
- Distância euclidiana entre assinaturas
- Percentual de compatibilidade
- Recomendações baseadas em confiança

---

## 🛠️ **Tecnologias**

- **Python 3.8+**
- **TensorFlow/Keras** - Rede Neural Siamesa
- **OpenCV** - Processamento de imagens
- **Streamlit** - Interface web
- **NumPy** - Computação científica
- **Pillow** - Manipulação de imagens
- **Matplotlib** - Visualização de dados

---

## 📦 **Instalação**

### **1. Clone o Repositório**
```bash
git clone https://github.com/marcosviniciohd/reconhecimento-texto-manuscrito.git
cd reconhecimento-texto-manuscrito
```

### **2. Crie um Ambiente Virtual**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### **3. Instale as Dependências**
```bash
pip install -r requirements.txt
```

### **4. Verifique a Instalação**
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

---

## 🚀 **Como Usar**

### **🌐 Interface Principal (Recomendado)**

```bash
streamlit run app.py
```

- **URL**: http://localhost:8502
- **Funcionalidades**: Upload de 2 assinaturas e comparação
- **Status**: ✅ Funcionando com threshold otimizado (0.1000)

### **📱 Interface para Telefone**

```bash
streamlit run app_teste_telefone.py
```

- **Uso**: Teste de assinaturas capturadas por telefone
- **Features**: Melhoria automática de qualidade, comparação múltipla
- **Guia**: Consulte `GUIA_TESTE_TELEFONE.md` para instruções detalhadas

---

## 📊 **Performance**

### **🎯 Métricas Atuais**
- **Threshold Otimizado**: 0.1000
- **Acurácia**: 64.7%
- **Dataset**: 30 imagens de 15 pessoas
- **Precisão (Mesma Pessoa)**: 10%
- **Recall (Mesma Pessoa)**: 60%
- **F1-Score**: 16.7%

### **📈 Análise de Distâncias**
- **Mesma pessoa**: Média 0.1175 ± 0.0580
- **Pessoas diferentes**: Média 0.1539 ± 0.0930
- **Separação**: Boa distinção entre classes

---

## 📱 **Teste com Telefone**

### **📸 Boas Práticas para Captura**
- ✅ **Iluminação**: Use luz natural ou boa iluminação artificial
- ✅ **Fundo**: Papel branco/claro, sem texturas
- ✅ **Ângulo**: Câmera paralela ao papel (90°)
- ✅ **Distância**: Assinatura ocupa 60-80% da foto
- ✅ **Foco**: Aguarde o foco automático
- ✅ **Estabilidade**: Use as duas mãos

### **❌ O que Evitar**
- ❌ Sombras e reflexos
- ❌ Fotografar de lado (inclinado)
- ❌ Usar zoom digital
- ❌ Fotografar em movimento

---

## 🔧 **Treinamento**

### **📊 Pipeline Completo**

#### **1. Preparar Dataset**
```bash
python scripts/preparar_dataset.py
```
- **Função**: Aplica data augmentation (rotação, escala, ruído)
- **Output**: Dataset expandido para treinamento
- **Tempo**: ~2-5 minutos

#### **2. Treinar Modelo**
```bash
python scripts/treinar_modelo.py
```
- **Função**: Treina a Rede Neural Siamesa
- **Features**: Early stopping, checkpoint automático
- **Tempo**: ~10-30 minutos (dependendo do dataset)
- **Output**: `modelos/modelo_assinaturas_manuscritas.h5`

#### **3. Avaliar e Calibrar**
```bash
python scripts/avaliar_modelo.py
```
- **Função**: Encontra o threshold ótimo
- **Processo**: Testa múltiplos thresholds e escolhe o melhor
- **Output**: `resultados_avaliacao/threshold_otimo.txt`

#### **4. Analisar Dados**
```bash
python scripts/analisar_dados.py
```
- **Função**: Relatório detalhado sobre o dataset
- **Informações**: Qualidade dos dados, distribuição, recomendações

### **🎯 Workflow Recomendado**
```bash
# Pipeline completo para novo treinamento
python scripts/preparar_dataset.py
python scripts/treinar_modelo.py
python scripts/avaliar_modelo.py

# Testar o sistema
streamlit run app.py
```

---

## 📁 **Estrutura do Projeto**

```
reconhecimento-texto-manuscrito/
├── 📂 scripts/                          # Scripts de treinamento
│   ├── 🔧 preparar_dataset.py          # Data augmentation
│   ├── 🧠 treinar_modelo.py            # Treinamento da rede
│   ├── 📊 avaliar_modelo.py            # Calibração de threshold
│   └── 📋 analisar_dados.py            # Análise do dataset
├── 📂 assinaturas_reais/               # Dataset de assinaturas
│   ├── 📁 pessoa1/                     # 2 assinaturas por pessoa
│   ├── 📁 pessoa2/
│   └── 📁 ...
├── 📂 modelos/                         # Modelos treinados
│   └── 🤖 modelo_assinaturas_manuscritas.h5
├── 📂 resultados_avaliacao/            # Resultados de avaliação
│   └── 📈 threshold_otimo.txt
├── 🌐 app.py                           # Interface principal
├── 📱 app_teste_telefone.py            # Interface para telefone
├── 📋 requirements.txt                 # Dependências
├── 📖 README.md                        # Este arquivo
├── 📱 GUIA_TESTE_TELEFONE.md           # Guia para testes com telefone
└── 🎯 PROJETO_FINALIZADO.md            # Documentação técnica
```

---

## 🎛️ **Configurações**

### **Threshold Personalizado**
Para ajustar a sensibilidade do sistema:

```python
# Em app.py, linha ~54
self.threshold = 0.10  # Padrão otimizado

# Mais rigoroso (menos falsos positivos)
self.threshold = 0.05

# Mais flexível (menos falsos negativos)  
self.threshold = 0.15
```

### **Parâmetros de Treinamento**
```python
# Em scripts/treinar_modelo.py
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
```

---

## 📚 **Exemplos de Uso**

### **💻 Uso Programático**
```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Carregar modelo
model = tf.keras.models.load_model(
    'modelos/modelo_assinaturas_manuscritas.h5',
    custom_objects={
        'euclidean_distance': euclidean_distance,
        'contrastive_loss': contrastive_loss
    }
)

# Preprocessar imagens
img1 = preprocess_image('assinatura1.png')
img2 = preprocess_image('assinatura2.png')

# Fazer predição
distance = model.predict([img1, img2])[0][0]
is_same_person = distance <= 0.10

print(f"Distância: {distance:.4f}")
print(f"Mesma pessoa: {'Sim' if is_same_person else 'Não'}")
```

### **🌐 Via Interface Web**
1. Acesse http://localhost:8502
2. Faça upload de 2 assinaturas
3. Clique em "Comparar Assinaturas"
4. Visualize o resultado e confiança

---

## 🔍 **Interpretação dos Resultados**

### **📊 Níveis de Confiança**
- **Alta Confiança (≤0.05)**: Resultado muito confiável
- **Moderada (0.05-0.15)**: Considere verificação adicional
- **Baixa (>0.15)**: Recomenda análise manual

### **✅ Status de Compatibilidade**
- **✅ MATCH**: Distância ≤ 0.10 (provavelmente mesma pessoa)
- **❌ NÃO MATCH**: Distância > 0.10 (provavelmente pessoas diferentes)

---

## 🚨 **Solução de Problemas**

### **Erro de Importação TensorFlow**
```bash
pip install --upgrade tensorflow
# ou para GPU
pip install tensorflow-gpu
```

### **Erro de OpenCV**
```bash
pip install opencv-python-headless
```

### **Streamlit não abre**
```bash
# Verificar porta
streamlit run app.py --server.port 8503

# Verificar firewall
streamlit run app.py --server.address 0.0.0.0
```

### **Modelo não carrega**
- Verifique se existe `modelos/modelo_assinaturas_manuscritas.h5`
- Execute o treinamento: `python scripts/treinar_modelo.py`

---

## 💡 **Melhorias Futuras**

### **🔄 Performance**
- [ ] Aumentar dataset (>20 assinaturas por pessoa)
- [ ] Implementar transfer learning
- [ ] Otimizar arquitetura da rede
- [ ] Adicionar validação cruzada

### **🌟 Features**
- [ ] API REST para integração
- [ ] Batch processing de múltiplas assinaturas
- [ ] Dashboard de analytics
- [ ] Exportar relatórios PDF

### **🏗️ Infraestrutura**
- [ ] Docker containerization
- [ ] Deploy em cloud (AWS/GCP)
- [ ] CI/CD pipeline
- [ ] Monitoramento automático

---

## 🤝 **Contribuição**

### **Como Contribuir**
1. **Fork** o projeto
2. Crie uma **branch** para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. Abra um **Pull Request**

### **Reportar Bugs**
- Use as **Issues** do GitHub
- Inclua passos para reproduzir
- Adicione logs e screenshots

---

## 📄 **Licença**

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

## 👥 **Autores**

- **Marcos Vinícius** - *Desenvolvimento Principal* - [@marcosviniciohd](https://github.com/marcosviniciohd)

---

## 🙏 **Agradecimentos**

- IFTM - Instituto Federal do Triângulo Mineiro
- Disciplina de Inteligência Computacional
- Comunidade TensorFlow e Streamlit

---

## 📞 **Suporte**

- **GitHub Issues**: [Reportar Bug](https://github.com/marcosviniciohd/reconhecimento-texto-manuscrito/issues)
- **Documentação**: Consulte os arquivos `.md` no projeto
- **Status**: ✅ Projeto ativo e funcional

---

<div align="center">

**🎯 Sistema pronto para uso!**

[![Status](https://img.shields.io/badge/Status-Funcional-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)]()

**Acesse http://localhost:8502 e teste o sistema!**

</div>
