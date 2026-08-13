# 🚗 Smart GAT

> Sistema inteligente de detecção, rastreamento e reconhecimento de placas veiculares utilizando Inteligência Artificial e Visão Computacional.

## 📌 Sobre o projeto

O **Smart GAT** é um projeto de visão computacional desenvolvido para detectar veículos e placas em vídeos, realizar o reconhecimento dos caracteres da placa e acompanhar os objetos detectados através de **tracking**.

O projeto combina **YOLO**, **OpenCV** e **EasyOCR**, criando uma base para um futuro sistema de monitoramento veicular inteligente.

## 🧠 Tecnologias

* 🐍 Python
* 🤖 YOLO11
* 👁️ OpenCV
* 🔤 EasyOCR
* 📦 Ultralytics
* 🎯 ByteTrack
* 🧠 PyTorch

## ⚙️ Funcionamento

O fluxo principal do Smart GAT é:

```text
📹 Vídeo / Câmera
       ↓
   OpenCV
       ↓
     YOLO
       ↓
 Detecção da placa
       ↓
    Tracking
       ↓
 Recorte da placa
       ↓
    EasyOCR
       ↓
 Reconhecimento do texto
```

## 🚘 Funcionalidades

### Detecção de placas

O modelo YOLO é treinado especificamente para localizar placas veiculares.

### 🔤 Reconhecimento de placas

Após a detecção, a região da placa é recortada e enviada para o **EasyOCR**, que tenta identificar os caracteres.

### 🎯 Rastreamento

O Smart GAT utiliza tracking para acompanhar objetos detectados entre os frames.

Cada objeto pode receber um identificador:

```text
ID: 1
ID: 2
ID: 3
```

Isso permite acompanhar o mesmo veículo durante sua movimentação no vídeo.

### 🎥 Processamento de vídeo

O sistema pode processar arquivos de vídeo utilizando OpenCV e possui estrutura para futuramente trabalhar com câmeras em tempo real.

## 📂 Estrutura do projeto

```text
smart-gat-AI/
│
├── app/
│   └── ai/
│       └── vehicle_detector.py
│
├── dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── video/
│
├── main.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🏋️ Treinamento

O modelo foi treinado utilizando um dataset de placas veiculares.

Exemplo de treinamento:

```bash
yolo detect train model=yolo11s.pt data=dataset/data.yaml epochs=150 imgsz=832 batch=8
```

Após o treinamento, o modelo gera os pesos em:

```text
runs/detect/train/weights/
```

O principal arquivo utilizado pelo sistema é:

```text
best.pt
```

> ⚠️ Os pesos do modelo e os resultados de treinamento não devem ser enviados ao GitHub caso sejam arquivos grandes. Mantenha-os no `.gitignore` ou utilize um sistema apropriado de armazenamento de modelos.

## ▶️ Como executar

### 1. Clone o projeto

```bash
git clone https://github.com/Ju1ninh0/smart-gat-AI.git
cd smart-gat-AI
```

### 2. Crie um ambiente virtual

Windows:

```powershell
python -m venv .venv
```

### 3. Ative o ambiente

```powershell
.venv\Scripts\activate
```

### 4. Instale as dependências

```powershell
pip install -r requirements.txt
```

### 5. Execute

```powershell
python main.py
```

## 📊 Exemplo

O sistema pode apresentar informações como:

```text
ID: 1
Placa: ABC1D23
Confiança: 0.94
```

Além de desenhar a região detectada diretamente no vídeo.

## 🚀 Próximos passos

O Smart GAT está em desenvolvimento. Algumas das próximas evoluções planejadas são:

* [x] Detecção de placas
* [x] Reconhecimento com OCR
* [x] Processamento de vídeo
* [x] Tracking de objetos
* [ ] Melhorar precisão do OCR
* [ ] Histórico de placas reconhecidas
* [ ] Banco de dados PostgreSQL
* [ ] Dashboard web
* [ ] Sistema de alertas
* [ ] Contagem de veículos
* [ ] Detecção de entrada e saída
* [ ] Suporte a câmeras IP/RTSP
* [ ] Análise de características do veículo
* [ ] Deploy em servidor
* [ ] Monitoramento em tempo real

## 🎯 Objetivo

O objetivo do Smart GAT é evoluir de um protótipo de visão computacional para uma plataforma completa de **monitoramento e análise inteligente de veículos**, combinando Inteligência Artificial, processamento de imagens, OCR, rastreamento e análise de dados.

## 👨‍💻 Desenvolvedor

**Ju1ninh0**

Estudante de Ciência da Computação interessado em:

* 🤖 Inteligência Artificial
* 🐍 Python
* 💻 Desenvolvimento de Software
* 🔐 Cybersecurity
* 👁️ Visão Computacional

---

⭐ Se o projeto for útil ou interessante, deixe uma estrela no repositório!
