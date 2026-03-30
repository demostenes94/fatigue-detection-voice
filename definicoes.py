import pyaudio

############################################################################################
#                     SVM Config
############################################################################################

# Nome do arquivo com modelo a ser usado na classificação das amostras de voz
MODEL_FILENAME = "melhor_modelo_rbf_3_classes.joblib"


############################################################################################
#                     Audio stream config
############################################################################################

# Formato dos dados retornados na caputura do audio
FORMAT = pyaudio.paFloat32

# Numero de canais de audio usados a captura (1 = mono)
CHANNELS = 1

# Taxa de amostragem para ser usada no VAD (Voice Activity Detector
SAMPLE_RATE_VAD = 16000  #  16 kHz, Silero VAD funciona somente com 16 kHz de taxa de amostragem

# Fator de multiplicação a ser usado na taxa de amostragem da captura de audio
# A taxa de 16 kHz necessária para o ilero VAD é muito baixa para a extração das features MFCC,
# por causa disto, usa-se um fator de 3 na captura
SAMPLE_RATE_FACTOR = 3

# Taxa de Amostragem de captura do audio, para extração dos MFCCs
SAPLE_RATE_CAPTURA = SAMPLE_RATE_VAD * SAMPLE_RATE_FACTOR # usa um sample rate 3 vezes maior que (48.000 kHz)

# Tamanho do frame para ser usado no Silero VAD
CHUNK_SIZE_SILERO_VAD = 512      # number of samples for each VAD call (~32ms)

# Tamanho do frame de audio na captura do microfone
CHUNK_SIZE = CHUNK_SIZE_SILERO_VAD * SAMPLE_RATE_FACTOR


# Pre-roll amount in ms (how much audio we include before "start" is triggered)
# For example, 100 ms → about 3 frames of 32 ms each
PRE_ROLL_MS = 100

# Tamanho do fram em milisegundos
FRAME_MS = (CHUNK_SIZE / SAPLE_RATE_CAPTURA) * 1000.0  # ~32 ms

# Quantidade de frames de pré-roll
NUM_PRE_ROLL_FRAMES = int(PRE_ROLL_MS // FRAME_MS) # ~3 frames


############################################################################################
#                     AVALIAÇÃO FADIGA Config
############################################################################################

# Quantidade de avalições a serem afetuadas antes de classificar o individou como Fadigado
NUMERO_AVALIACOES = 50

# tamanho do segmento de audio a ser avaliado ~2.000 ms
AUDIO_SEGMENT_MS = 2000

# Numero de frames de audio necessário para formar 2 segundos de audio
NUM_AUDIO_SEGMENT_FRAMES = int(AUDIO_SEGMENT_MS // FRAME_MS)  # ~62 frames


############################################################################################
#                     Voice Probability config
############################################################################################

# Sensibilidade para detectar voz (valores entre 0.1 e 1.00), um valor maior irá deixar a detecção menos sensivel
VOICE_PROB = 0.2