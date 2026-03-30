# 🎙️ Detecção de Fadiga por Voz

Projeto de Machine Learning para detecção de fadiga baseado na análise de voz em tempo real.

---

## 🚀 Objetivo

Desenvolver um sistema capaz de identificar sinais de fadiga através da voz, com aplicação em ambientes críticos como controle de tráfego aéreo.

---

## 🧠 Pipeline do Projeto

* Captura de áudio em tempo real via microfone
* Detecção de voz (VAD - Voice Activity Detection)
* Extração de características acústicas (MFCC)
* Classificação com modelo SVM
* Visualização em tempo real dos resultados

---

## 🛠️ Tecnologias utilizadas

* Python
* NumPy / Pandas
* Librosa
* PyAudio
* Scikit-learn
* Matplotlib
* PyTorch (Silero VAD)

---

## 📊 Resultados

O modelo atingiu até **96,47% de acurácia** na detecção de fadiga.

---

## ▶️ Como executar

```bash
pip install -r requirements.txt
python DeteccaoFadiga_v5.py
```

---

## 📌 Observações

* O sistema utiliza processamento em tempo real com múltiplas threads
* Necessário microfone configurado no sistema
* O modelo treinado deve estar disponível localmente

---

## 👨‍💻 Autores

Demóstenes Ramos
Adriano Luis Bruch

