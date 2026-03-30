# Biblioteca para armazenar as funções utilizadas no programa

import numpy as np
import queue
from collections import deque
import time
import pyaudio
import torch
from scipy import signal
import librosa
import pandas as pd
import joblib

import definicoes

# Fila para armazenar os frames de audio capturados pelo microfone
microfone_audio_queue = queue.Queue()

# Fila com os frames de audio com voz para extracao das features
features_audio_queue = queue.Queue()

# Fila com os coeficientes MFCC estraídos das amostras de audio
features_mfcc_queue = queue.Queue()

# Fila circular para armazenar as últimas N classificações instatâneas de fadiga
ring_ultimas_avaliacaoes = deque(maxlen=definicoes.NUMERO_AVALIACOES)

# Fila circular para armazenar as últimas N probabilidades de fadiga
ring_probabilidade_fadiga = deque(maxlen=definicoes.NUMERO_AVALIACOES)

#----------------------------------------------------------------
# Função para mostrar uma barra do probabilidade no console
#----------------------------------------------------------------
def visualize_bar(value):
    # Ensure value is within 0 to 1
    value = max(0, min(1, value))
    total_length = 20
    filled_length = int(round(value * total_length))
    bar = '█' * filled_length + '-' * (total_length - filled_length)
    return f"[{bar}] {value:.2f}"

#----------------------------------------------------------------
# Função para identificar se a probabilidade é voz
# A função usa como limiar de decisão o valor definido em definicoes.VOICE_PROB
#----------------------------------------------------------------
def is_speech(probability):
    
    # Se a probalidade for maior que o limiar, classifica como voz
    if probability > definicoes.VOICE_PROB:
        return True
    else:
        return False

#----------------------------------------------------------------
# Função para calcular a probabilidade do frame de audio ser voz
#
#   voice_probability: Retorna a probabilidade do frame ser audio
#----------------------------------------------------------------
def speech_probability(vad_model, audio_chunk):
    
    # Converte o frame de audio de binário para um array numpy de float32
    audio_np = np.frombuffer(audio_chunk, dtype=np.float32)

    #Faz a decimação do audio, para a taxa de amostragem aceita pelo Silero  (16 kHz)
    resampled_audio_np = signal.decimate(audio_np, definicoes.SAMPLE_RATE_FACTOR)

    # Converte para tensor usando a amostra decimada
    audio_tensor = torch.from_numpy(np.copy(resampled_audio_np) )

    # Calcula a probabilidade de ser voz
    with torch.no_grad():
        voice_probability = vad_model(audio_tensor, definicoes.SAMPLE_RATE_VAD).item()

    # retorna a probabilidade de ser voz e um array numpy com a amostra de voz
    return voice_probability

#----------------------------------------------------------------
# Função para extrair os coeficientes MFCC da amostra de audio
#   Parametros:
#       y: array numpy (float32) com audio a ser processado
#       sr: taxa de amostragem a ser utlizada na extração dos coeficientes
#       n_mfcc: número de coeficientes MFCC a serem extraidos
#   Retorna:
#       df: DataFrame Pandas com as features extraídas (média, desvio padrão, minimo, máximo e range)
#----------------------------------------------------------------
def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = {}

    for i in range(n_mfcc):
        coef = mfcc[i]
        prefix = f"mfcc[{i}]"
        features[f"{prefix}_amean"] = np.mean(coef)
        features[f"{prefix}_stddev"] = np.std(coef)
        features[f"{prefix}_min"] = np.min(coef)
        features[f"{prefix}_max"] = np.max(coef)
        features[f"{prefix}_range"] = np.max(coef) - np.min(coef)
        
    dados = []
    dados.append(features)
    df = pd.DataFrame(dados)
    return df


#-----------------------------------------------------------------------
# Thread function Responsável pela captura do audio pelo microfone
#-----------------------------------------------------------------------
def microfone_reader_thread():
    
    # Incializa PyAudio para captura do audio via microfone
    p = pyaudio.PyAudio()

    # Inicia a captura de audio do microfone
    stream = p.open(format=definicoes.FORMAT,
                    channels=definicoes.CHANNELS,
                    rate=definicoes.SAPLE_RATE_CAPTURA,
                    input=True,
                    frames_per_buffer=definicoes.CHUNK_SIZE)

    try:
        # Loop para manter a thread ativa
        while True:
            # Captura o audio em frames de 1536 amostras
            data = stream.read(definicoes.CHUNK_SIZE, exception_on_overflow=False)
            # coloca o frame capturado na fila de detecção de voz
            microfone_audio_queue.put(data)
    
    finally:
        # Libera o microne ao finalizar a thread
        stream.stop_stream()
        stream.close()
        p.terminate()


#-----------------------------------------------------------------------
# Thread function Responsável pela detecção de fala
#-----------------------------------------------------------------------
def voice_detection_thread():

    # Carrega o modelo Silero VAD para detecacao de voz
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Buffer para armazenar os últimos 3 frames antes da detecção de voz
    ring_buffer = deque(maxlen=definicoes.NUM_PRE_ROLL_FRAMES)
    
    # Buffers for speech segment detection
    audio_buffer = bytearray() 
    audio_buffer_np = np.zeros(1, dtype=np.float32)

    # Variável usada para identificar quando detectou voz
    triggered = False
    
    # Contador com o numero de frames de voz detectados
    num_frames_voz = 0
    
    # contador de frames após não identificar mais voz
    num_after_frames = 0

    # Loop para manter a thread ativa
    while True:
        
        # Se houver algum frame na fila esperando detecção de voz
        if not microfone_audio_queue.empty():
            
            # Retira um frame de audio binário da fila do microfone
            audio_chunk = microfone_audio_queue.get()

            # Verifica se o frame é voz e converte de binário para uma matriz numpy
            voice_probability = speech_probability(vad_model, audio_chunk)
            
            # É voz, mas a gravação não estava acionada
            if is_speech(voice_probability) and (not triggered):
                
                #Adiciona os 100 ms anteriores ao buffer de voz
                for rb_chunk in ring_buffer:
                    audio_buffer.extend(rb_chunk)
                ring_buffer.clear()
                
                # Sinaliza que detectou voz
                triggered = True
                print("Detectou voz")
            
            # Se é voz, armazena o frame no buffer de voz
            if triggered:
                # Adiciona o frame no buffer binário de audio
                audio_buffer.extend(audio_chunk)

                # Converte o frame de audio de binário para um array numpy de float32
                audio_np = np.frombuffer(audio_chunk, dtype=np.float32)
                # Adiciona o frame de voz buffer numpy de audio
                audio_buffer_np = np.append(audio_buffer_np, audio_np)
                
                #Incrementa a contagem de frames de voz dectados
                num_frames_voz = num_frames_voz + 1
                
            
            # Se a gravação não estiver ativa, armazena o frame no buffer circular
            else:
                ring_buffer.append(audio_chunk)

            # Estava com a gravação ativa, mas detectou um frame sem voz, incrementa a contagem de frames sem voz
            if (not is_speech(voice_probability)) and triggered:
                num_after_frames = num_after_frames + 1

            #Para a gravação após 3 frames sem voz
            if num_after_frames > definicoes.NUM_PRE_ROLL_FRAMES:
                num_after_frames = 0
                triggered = False
            
            # Detectou 62 frames de voz, apraximadamente 2.000 ms, envio o bloco de 
            # audio para a fila de extração de features
            if num_frames_voz >= definicoes.NUM_AUDIO_SEGMENT_FRAMES:
                features_audio_queue.put(audio_buffer_np)
                num_frames_voz = 0
            
            #print(f"{visualize_bar(voice_probability)} {features_audio_queue.qsize()} / {len(audio_buffer)}")
        
        # Não amostras na fila para extração de features, dorme por 0.1 segundos
        else:
            time.sleep(0.1)    

#-----------------------------------------------------------------------
# Thread function Responsável pela extração das features (MFCCs) das amostras de audio
#-----------------------------------------------------------------------
def extract_mfcc_thread():

    # Loop para manter a thread ativa
    while True:
        # Se houver alguma amostra na fila esperando extração de features
        if not features_audio_queue.empty():
            # Retira a amostra da fila
            audio_chunk_feature = features_audio_queue.get()
            # Extrai as features
            mean_mfcc = extract_mfcc(audio_chunk_feature, definicoes.SAPLE_RATE_CAPTURA, 13)
            # Coloca as features extraidas na fila de features esperando classificação
            features_mfcc_queue.put(mean_mfcc)
        
        # Não amostras na fila para extração de features, dorme por 0.2 segundos
        else:
            time.sleep(0.2)
                
#-----------------------------------------------------------------------
# Thread function Responsável pela classificação em fadigado/vigilia
#-----------------------------------------------------------------------
def classificacao_thread():

    # Carrega o Modelo de Classificação SVM
    modelo_svm = joblib.load(definicoes.MODEL_FILENAME)

    #Limpa as últimas avaliações
    ring_ultimas_avaliacaoes.clear()
    for i in range(definicoes.NUMERO_AVALIACOES):
        ring_ultimas_avaliacaoes.append(0)

    # Loop para manter a thread ativa
    while True:
        
        # Se houver alguma feature na fila esperando classificação
        if not features_mfcc_queue.empty():
            # Retira a feature da fila
            mfcc = features_mfcc_queue.get()
            
            # Faz a predição usando o modelo previamento treinado
            predicao = modelo_svm.predict(mfcc)

            # Adiciona o resultado da predição nas últimas avaliação (obs.: .item() converte para escalar)
            ring_ultimas_avaliacaoes.append(predicao.item())

            # Calcula a média das últimas N avaliações
            ultimas_avaliacoes_np = np.array(ring_ultimas_avaliacaoes)
            media = np.mean(ultimas_avaliacoes_np) * 100

            # Adiciona a média na fila circular
            ring_probabilidade_fadiga.append(media.item())
            
            # Classifica a predição entre vigilia, indefinido a fadigado
            if predicao == -1: 
                print(f"Predicao = {predicao.item()} (Vigilia) / Media: {media}")
            elif predicao == 0: 
                print(f"Predicao = {predicao.item()} (Indefinido) / Media: {media}")
            else:
                print(f"Predicao = {predicao.item()} (Fadigado) / Media: {media}")

            #print(f"ring_ultimas_avaliacaoes: {list(ring_ultimas_avaliacaoes)}")
            #print(f"Média das últimas avaliações {np.mean(list(ring_ultimas_avaliacaoes))}")


        # Não havia features na fila de classificação, dorme por 0.2 segundos
        else:
            time.sleep(0.2)