# Guia de Execução Local

Este guia contém todas as etapas para configurar e executar o projeto de verificação de assinaturas.

---

## 1. Requisitos de Ambiente

- **Python 3.10 ou 3.11 (64 bits)**
  - O TensorFlow, uma das principais dependências, não é compatível com Python 3.12 ou superior.
  - Baixe a versão correta em: [Python 3.10.0](https://www.python.org/downloads/release/python-3100/)

---

## 2. Configuração do Ambiente Virtual

O uso de um ambiente virtual é essencial para isolar as dependências do projeto.

1.  **Navegue até a pasta raiz do projeto** (a pasta que contém o arquivo `requirements.txt`).
2.  **Crie o ambiente virtual**:
    ```bash
    python -m venv .venv
    ```
3.  **Ative o ambiente**:
    - No Git Bash:
      ```bash
      source .venv/Scripts/activate
      ```
    - No PowerShell:
      ```powershell
      .venv\Scripts\Activate.ps1
      ```
    - No Prompt de Comando (CMD):
      ```cmd
      .venv\Scripts\activate
      ```

---

## 3. Instalação das Dependências

Com o ambiente virtual ativado, instale todas as bibliotecas necessárias a partir do arquivo `requirements.txt` que está na pasta raiz:

```bash
# Se você estiver na pasta /src, suba um nível para a raiz
pip install -r ../requirements.txt
```

---

## 4. Como Executar a Aplicação

Após a instalação, você pode treinar o modelo e executar a aplicação a partir da pasta `src`.

### Treinar o Modelo

Se o arquivo `siamese_signature_model.h5` ainda não existir, você precisa treiná-lo primeiro.

```bash
# Estando na pasta /src
python train.py
```

Isso irá gerar o arquivo `siamese_signature_model.h5` neste mesmo diretório.

### Executar a Interface Gráfica

Para iniciar a aplicação web e comparar assinaturas:

```bash
# Estando na pasta /src
streamlit run app.py
```

Após executar o comando, uma aba no seu navegador será aberta no endereço `http://localhost:8501`. Siga as instruções na tela para fazer o upload das imagens e verificar a compatibilidade.
