import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

# Variáveis para armazenar os dados
X = []
Y = []
Z = []
S = []
T = []

# Verificar se o nome do arquivo foi fornecido como argumento
if len(sys.argv) < 2:
    print("Por favor, forneça o nome do arquivo como argumento.")
    sys.exit(1)

# Nome do arquivo fornecido como argumento
nome_arquivo = sys.argv[1]

# Verificar se o arquivo existe
if not os.path.exists(nome_arquivo):
    print(f"O arquivo '{nome_arquivo}' não existe.")
    sys.exit(1)
filename = nome_arquivo
# Abre o arquivo CSV em modo de leitura
with open(filename, 'r') as arquivo_csv:
    leitor_csv = csv.reader(arquivo_csv)

    # Itera pelas linhas do arquivo CSV
    for linha in leitor_csv:
        # Separa os elementos em cada linha e converte para float ou int
        elementos = linha[0].split(';')
        x = float(elementos[0].strip())
        y = float(elementos[1].strip())
        z = float(elementos[2].strip())
        s = int(elementos[3].strip())
        t = int(elementos[4].strip())

        X.append(x)
        Y.append(y)
        Z.append(z)
        S.append(s)
        T.append(t)

intervalo_ms = 15000  # 30 segundos em milissegundos

num_intervalos = int(len(T) / intervalo_ms)

X_split =np.array_split(X, num_intervalos)
Y_split =np.array_split(Y, num_intervalos)
Z_split =np.array_split(Z, num_intervalos)
T_split =np.array_split(T, num_intervalos)

nperseg = 1024  # Número de pontos por segmento
noverlap = nperseg // 2  # Sobreposição entre segmentos

for i in range(num_intervalos):
    intervaloX = X_split[i]
    intervaloY = Y_split[i]
    intervaloZ = Z_split[i]
    tempoIntervalo = T_split[i]

    fs = 1.0 / (tempoIntervalo[1] - tempoIntervalo[0])
    
     # Plotar gráfico do intervalo
    plt.specgram(intervaloX, NFFT=nperseg, Fs=fs, noverlap=noverlap, cmap='inferno')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Espectro de Frequência - Eixo X - Intervalo {i+1}')
    plt.grid(True)
    pasta_destino = f'{filename}_eixo_x'
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    # Salvar gráfico em arquivo
    nome_arquivo = f'{pasta_destino}/grafico_intervalo_{i+1}_x.png'
    plt.savefig(nome_arquivo)

    # Limpar figura para o próximo intervalo
    plt.clf()

    # Plotar gráfico do intervalo
    plt.specgram(intervaloY, NFFT=nperseg, Fs=fs, noverlap=noverlap, cmap='inferno')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Espectro de Frequência - Eixo Y - Intervalo {i+1}')
    plt.grid(True)

    pasta_destino = f'{filename}_eixo_y'
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
    # Salvar gráfico em arquivo
    nome_arquivo = f'{pasta_destino}/grafico_intervalo_{i+1}_Y.png'
    plt.savefig(nome_arquivo)

    # Limpar figura para o próximo intervalo
    plt.clf()

    # Plotar gráfico do intervalo
    plt.specgram(intervaloZ, NFFT=nperseg, Fs=fs, noverlap=noverlap, cmap='inferno')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Espectro de Frequência - Eixo Z - Intervalo {i+1}')
    plt.grid(True)

    pasta_destino = f'{filename}_eixo_z'
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
    # Salvar gráfico em arquivo
    nome_arquivo = f'{pasta_destino}/grafico_intervalo_{i+1}_Z.png'
    plt.savefig(nome_arquivo)

    # Limpar figura para o próximo intervalo
    plt.clf()

    print(f"Gráfico do intervalo {i+1} salvo em {nome_arquivo}")

