# Smart GAT

Sistema de visão computacional para detecção, rastreamento e reconhecimento de placas veiculares utilizando Inteligência Artificial.

## Sobre

O **Smart GAT** é um sistema de visão computacional desenvolvido em Python para detecção e monitoramento de veículos e placas.

O projeto combina **YOLO11, OpenCV, ByteTrack e EasyOCR** com uma aplicação web desenvolvida em **Flask**, permitindo processar vídeos, rastrear veículos, reconhecer placas e visualizar os dados através de um dashboard.

## Tecnologias

* Python
* YOLO11
* Ultralytics
* OpenCV
* EasyOCR
* ByteTrack
* PyTorch
* Flask
* SQLite

## Pipeline

```text
Vídeo / Câmera
      ↓
    OpenCV
      ↓
    YOLO11
      ↓
Detecção de veículos
      ↓
   ByteTrack
      ↓
 Rastreamento
      ↓
Detecção da placa
      ↓
   EasyOCR
      ↓
Reconhecimento
      ↓
 Banco de dados
      ↓
 Dashboard
```

## Funcionalidades

### Visão Computacional

* [x] Detecção de veículos
* [x] Detecção de placas
* [x] Rastreamento de veículos
* [x] Identificação por ID
* [x] Reconhecimento de placas com OCR
* [x] Processamento de vídeos
* [x] Exibição das detecções no vídeo
* [x] Registro das informações detectadas

### Dashboard

* [x] Dashboard web com Flask
* [x] Monitoramento de veículos
* [x] Visualização de placas reconhecidas
* [x] Gerenciamento de câmeras
* [x] Página de veículos
* [x] Página de placas
* [x] Página de câmeras
* [x] Estatísticas de monitoramento
* [x] Interface web responsiva
* [x] Navegação entre módulos

### Dados

* [x] Armazenamento das detecções
* [x] Registro de veículos
* [x] Registro de placas
* [x] Registro de câmeras
* [x] Histórico de informações
* [x] Integração com banco de dados

### Em desenvolvimento

* [ ] Melhorar precisão do OCR
* [ ] PostgreSQL
* [ ] Sistema de alertas
* [ ] Contagem de veículos
* [ ] Detecção de entrada e saída
* [ ] Suporte a câmeras IP / RTSP
* [ ] Análise de características dos veículos
* [ ] Monitoramento em tempo real
* [ ] Deploy em servidor

## Estrutura

```text
smart-gat-AI/
│
├── app/
│   ├── ai/
│   │   └── vehicle_detector.py
│   │
│   └── dashboard/
│       ├── app.py
│       ├── templates/
│       │   ├── dashboard.html
│       │   ├── cameras.html
│       │   ├── plates.html
│       │   └── vehicles.html
│       └── static/
│
├── data/
├── dataset/
├── video/
│
├── main.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Treinamento

O modelo YOLO11 foi treinado utilizando um dataset específico para detecção de placas veiculares.

```bash
yolo detect train model=yolo11s.pt data=dataset/data.yaml epochs=150 imgsz=832 batch=8
```

O peso principal utilizado para inferência é:

```text
best.pt
```

## Instalação

```bash
git clone https://github.com/Ju1ninh0/smart-gat-AI.git
cd smart-gat-AI
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute:

```bash
python main.py
```

Para iniciar o dashboard:

```bash
python app/dashboard/app.py
```

O dashboard estará disponível localmente em:

```text
http://127.0.0.1:5000
```

## Objetivo

O objetivo do Smart GAT é evoluir de um protótipo de visão computacional para uma plataforma completa de **monitoramento e análise inteligente de veículos**.

A arquitetura combina inteligência artificial, processamento de imagens, OCR, tracking, banco de dados e uma interface web para centralizar as informações coletadas.

## Autor

**Ju1ninh0**

⭐ Se o projeto for interessante, considere deixar uma estrela no repositório.
