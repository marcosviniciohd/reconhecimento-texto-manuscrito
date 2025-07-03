

# Verificação de Assinaturas Manuscritas com Redes Neurais Siamesas

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/tensorflow-%3C2.16-important" alt="TensorFlow Version">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## 📑 Sumário

- [Descrição do Projeto](#descrição-do-projeto)
- [Requisitos de Ambiente](#requisitos-de-ambiente)
- [Como criar um ambiente virtual com Python 3.10](#como-criar-um-ambiente-virtual-com-python-310)
- [Instalação das Dependências](#instalação-das-dependências)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Estrutura Recomendada de Pastas para o Dataset](#estrutura-recomendada-de-pastas-para-o-dataset)
- [Exemplos de Caminhos para Imagens](#exemplos-de-caminhos-para-imagens)
- [Como Modificar o Código para Usar Imagens Reais](#como-modificar-o-código-para-usar-imagens-reais)
- [Listar Arquivos do Dataset no Notebook (Exemplo)](#listar-arquivos-do-dataset-no-notebook-exemplo)
- [Dicas Importantes](#dicas-importantes)
- [Passos para Rodar o Projeto](#passos-para-rodar-o-projeto)
- [Funcionalidades Implementadas](#funcionalidades-implementadas)
- [Próximos Passos Sugeridos](#próximos-passos-sugeridos)
- [FAQ - Problemas Comuns](#faq---problemas-comuns)
- [Sobre o Grupo](#sobre-o-grupo)


## Descrição do Projeto

Este projeto acadêmico tem como objetivo desenvolver uma aplicação de Deep Learning para verificar a autenticidade de assinaturas manuscritas. O sistema será capaz de comparar duas imagens de assinaturas e determinar se elas pertencem à mesma pessoa, utilizando **Redes Neurais Siamesas (Siamese Neural Networks)**.

---

## Requisitos de Ambiente

- **Python 3.10 ou 3.11 (64 bits)**
  - O TensorFlow não é compatível com Python 3.12 ou superior.
  - Baixe em: https://www.python.org/downloads/release/python-3100/

## Como criar um ambiente virtual com Python 3.10

1. Instale o Python 3.10 e adicione ao PATH durante a instalação.
2. No terminal, navegue até a pasta do projeto e execute:
   ```bash
   python -m venv .venv
   ```
   Ou especifique o caminho completo do Python 3.10, se necessário:
   ```bash
   "C:/Users/SeuUsuario/AppData/Local/Programs/Python/Python310/python.exe" -m venv .venv
   ```
3. Ative o ambiente virtual:
   - No Bash:
     ```bash
     source .venv/Scripts/activate
     ```
   - No Prompt de Comando:
     ```cmd
     .venv\Scripts\activate
     ```
   - No PowerShell:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

## Instalação das Dependências

Com o ambiente virtual ativado, execute:
```bash
pip install -r requirements.txt
```
Se for usar a interface gráfica, instale também o Streamlit (caso não esteja no requirements):
```bash
pip install streamlit
```

---

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

- **/data**: Diretório para armazenar o dataset de assinaturas.
- **/src**: Código-fonte da aplicação.
  - `data_preprocessing.py`: Funções para pré-processamento das imagens.
  - `model.py`: Definição da arquitetura da Rede Neural Siamesa.
  - `train.py`: Script para treinar o modelo.
  - `evaluate.py`: Script para avaliar o modelo treinado.
  - `app.py`: Interface gráfica para testar o modelo.
- `requirements.txt`: Lista de dependências Python do projeto.
- `notebooks/`: Jupyter Notebooks para exploração de dados e apresentação dos resultados.

---

## Estrutura Recomendada de Pastas para o Dataset

```
data/
  ├── autor1/
  │     ├── assinatura1.png
  │     ├── assinatura2.png
  │     └── ...
  ├── autor2/
  │     ├── assinatura1.png
  │     └── ...
  └── ...
```

## Exemplos de Caminhos para Imagens

- data/autor1/assinatura1.png
- data/autor2/assinatura2.jpg

## Como Modificar o Código para Usar Imagens Reais

No notebook ou nos scripts, altere as variáveis de caminho, por exemplo:
```python
data_dir = 'data'
img_path1 = os.path.join(data_dir, 'autor1', 'assinatura1.png')
img_path2 = os.path.join(data_dir, 'autor2', 'assinatura2.png')
```

## Listar Arquivos do Dataset no Notebook (Exemplo)

```python
import os
for autor in os.listdir('data'):
    print(f"Autor: {autor}")
    for img in os.listdir(os.path.join('data', autor)):
        print(f"  - {img}")
```

## Dicas Importantes

- As imagens podem ser coloridas, mas o código faz a conversão automática para tons de cinza.
- Se necessário, ajuste o pré-processamento (tamanho, binarização) em `src/data_preprocessing.py` ou no notebook.
- Certifique-se de que os nomes das pastas e arquivos não tenham espaços ou caracteres especiais.
- O modelo treinado deve estar em `src/siamese_signature_model.h5` para uso na interface.

---

## Passos para Rodar o Projeto

1. **Instale as dependências** (veja acima).
2. **Treine o modelo** (se ainda não existe o arquivo `siamese_signature_model.h5`):
   ```bash
   python src/train.py
   ```
3. **Execute a interface gráfica para comparar assinaturas:**
   ```bash
   streamlit run src/app.py
   ```
4. O navegador será aberto automaticamente. Siga os passos na tela:
   - Faça upload da primeira imagem de assinatura.
   - Faça upload da segunda imagem de assinatura.
   - O sistema irá mostrar as imagens, a distância calculada e o resultado (compatíveis ou diferentes).

---

## Funcionalidades Implementadas

- Pré-processamento de imagens: conversão para escala de cinza, binarização, redimensionamento e normalização.
- Criação de pares de imagens (genuína/genuína e genuína/falsa) para treinamento.
- Arquitetura de Rede Neural Siamesa baseada em CNN.
- Função de custo Contrastive Loss para aprendizado de similaridade.
- Treinamento do modelo e salvamento do modelo treinado.
- Avaliação do modelo com métricas: acurácia, precisão, recall, F1-score e matriz de confusão.
- Interface gráfica para teste de assinaturas (Streamlit).
- Notebooks para apresentação dos resultados e explicação detalhada do código.

---

## Próximos Passos Sugeridos

- Adicionar exemplos de uso com datasets reais no notebook.
- Preparar slides e divisão de tópicos para apresentação igualitária entre os membros do grupo.

---

## FAQ - Problemas Comuns

**Q: Recebo erro de versão do TensorFlow ou Python ao instalar as dependências.**

A: Certifique-se de estar usando Python 3.10 ou 3.11 (64 bits). O TensorFlow não suporta Python 3.12+. Veja a seção [Requisitos de Ambiente](#requisitos-de-ambiente).

**Q: Não consigo ativar o ambiente virtual no Windows.**

A: Se aparecer erro de permissão, execute o PowerShell como administrador e rode:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Depois tente ativar novamente.

**Q: O navegador não abre automaticamente ao rodar o Streamlit.**

A: Acesse manualmente http://localhost:8501 no navegador.

**Q: O modelo treinado não é encontrado ao abrir a interface.**

A: Certifique-se de ter rodado o treinamento (`python src/train.py`) e que o arquivo `src/siamese_signature_model.h5` existe.

**Q: Outros erros?**

A: Consulte as mensagens do terminal e verifique se todas as dependências estão instaladas corretamente.

---

## Sobre o Grupo

Projeto desenvolvido por alunos do curso de Ciência da Computação do IFTM.

- Integrantes: [NOME1], [NOME2], [NOME3], [NOME4] <!-- Substitua pelos nomes reais -->
- Orientador: [Nome do Professor]
- Contato: [e-mail ou GitHub do grupo]

---
