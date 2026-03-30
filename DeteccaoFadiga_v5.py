import threading
import time
import numpy as np
from numpy import interp
from numpy import pi
import csv
from datetime import datetime
import os
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.text

# Bibliotecas locais
import biblioteca
import definicoes

# --------------------- CSV ---------------------
data_inicio = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nome_arquivo = f"resultados_{data_inicio}.csv"

with open(nome_arquivo, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["data_hora", "resultado"])

# Função para salvar avaliação no CSV
def salvar_avaliacao_csv(resultado):
    data_hora_agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(nome_arquivo, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data_hora_agora, resultado])

# --------------------- Threads ---------------------
def iniciar_threads():
    threading.Thread(target=biblioteca.microfone_reader_thread, daemon=True).start()
    threading.Thread(target=biblioteca.voice_detection_thread, daemon=True).start()
    threading.Thread(target=biblioteca.extract_mfcc_thread, daemon=True).start()
    threading.Thread(target=biblioteca.classificacao_thread, daemon=True).start()
    atualizar_grafico()

# --------------------- GUI ---------------------
root = tk.Tk()
root.title("Detecção de Fadiga por Voz")

# Botões
frame_botoes = ttk.Frame(root)
frame_botoes.pack(pady=10)

btn_iniciar = ttk.Button(frame_botoes, text="Iniciar Gravação", command=iniciar_threads)
btn_iniciar.pack(side=tk.LEFT, padx=5)

btn_parar = ttk.Button(frame_botoes, text="Parar Gravação", command=root.quit)
btn_parar.pack(side=tk.LEFT, padx=5)

# Gráfico
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
ax0.remove()
ax0 = fig.add_subplot(1, 3, 1, projection="polar")

def formatar_graficos(ax0, ax1, ax2, media):

    # Gauge chart
    colors = ["forestgreen", "limegreen", "chartreuse", "yellowgreen", "yellow", "gold", "orange", "darkorange", "orangered", "red"]
    values = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0] 
    values = [100, 80, 60, 40, 20, 0, -20, -40, -60, -80, -100] 
    bars = np.linspace(-np.pi/4, np.pi*5/4, 10, endpoint=False)
    values_pos = np.linspace(-np.pi/4, np.pi*5/4, 11, endpoint=True)
    
    ax0.bar(x=bars, width=0.5, height=0.5, bottom=2,
       linewidth=3, edgecolor="white",
       color=colors, align="edge");
    
    #mapeia o valor percentual em escala polar 
    posicao = interp(media, [-100,100], [np.pi*5/4, -np.pi/4])

    # Get a list of all Annotation objects
    annotations = [child for child in ax0.get_children() if isinstance(child, matplotlib.text.Annotation)]
    
    for i, annotation in enumerate(annotations):
        annotation.remove()

    for loc, val in zip(values_pos, values):
        plt.annotate(val, xy=(loc, 2.5), ha="right" if val<=50 else "left",)

    plt.annotate(int(media), xytext=(0,0), xy=(posicao, 2.0),
            arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black", shrinkA=0),
            bbox=dict(boxstyle="circle", facecolor="black", linewidth=2.0, ),
            fontsize=25, color="white", ha="center"
            );
    
    plt.title("Fadiga", loc="center");
    ax0.set_axis_off()

    
    # Grafico da evolucao das probabilidades
    ax1.set_title('Probabilidade de Fadiga')
    #ax1.set_xlabel('avaliações')
    #ax1.set_ylabel('%')
    ax1.set_xlim(0, definicoes.NUMERO_AVALIACOES)
    ax1.set_ylim(-105, 105)
    ax1.grid(True, axis='y')

    # Grafico das avaliações instantâneas
    ax2.set_title('Avaliações Instantâneas')
    #ax2.set_xlabel('Número de avaliações')
    #ax2.set_ylabel('Resultado')
    ax2.set_xlim(0, definicoes.NUMERO_AVALIACOES)
    ax2.set_ylim(-1.1, 1.1)
    custom_yticks_ax2 = [-1, 0, 1]
    custom_ylabels_ax2 = ['Vigília', 'Indefinido', 'Fadigado']
    ax2.set_yticks(custom_yticks_ax2)
    ax2.set_yticklabels(custom_ylabels_ax2, ha='right')
    ax2.grid(True, axis='y')    


# ax0 é o gauge chart
#fadiga_gauge_chart(ax0, 0)
# ax1 é o gráfico da evolução da fadiga, ax2 são as avaliações instantâneas
formatar_graficos(ax0, ax1, ax2, 0)


canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
canvas.draw()
plt.tight_layout()



# Atualiza gráfico em tempo real
def atualizar_grafico():
    try:
        ultimas_avaliacoes = np.array(biblioteca.ring_ultimas_avaliacaoes)
        if len(ultimas_avaliacoes) > 0:
            # Salva a última avaliação no CSV
            salvar_avaliacao_csv(ultimas_avaliacoes[-1])
        
        # Plota o gráfico das últimas avaliações instaneas
        ax2.clear()
        if len(ultimas_avaliacoes) > 0:
            ax2.stem(ultimas_avaliacoes.reshape(-1), linefmt='green')

        # Plota o gráfico da média das ultimas avaliações
        probabilidades_fadiga = np.array(biblioteca.ring_probabilidade_fadiga)
        x = np.arange(len(probabilidades_fadiga))

        ax1.clear()
        if len(probabilidades_fadiga) > 0:
            ax1.step(x, probabilidades_fadiga.flatten(), color='g')
            ax1.plot(x, probabilidades_fadiga.flatten(), 'o', markersize=2, color='g')
            media = probabilidades_fadiga[-1]
        else:
            media = 0;

        formatar_graficos(ax0, ax1, ax2, media)

        canvas.draw()

    except Exception as e:
        print(f"Erro ao atualizar gráfico: {e}")
    
    # Chama novamente depois de 1s
    root.after(1000, atualizar_grafico)

# Mensagem inicial
label_info = ttk.Label(root, text=f"Para ser classificado como fadigado, a média das últimas {definicoes.NUMERO_AVALIACOES} avaliações deve ser > 0.5")
label_info.pack(pady=10)

root.mainloop()
