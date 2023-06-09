import numpy as np
import matplotlib.pyplot as plt
import csv

# Variáveis para armazenar os dados
X = [][10]
Y = [][10]
Z = [][10]
S = [][10]
T = [][10]

# Abre o arquivo CSV em modo de leitura
with open('accel_80_F6.csv', 'r') as arquivo_csv:
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

intervalo_ms = 30000  # 30 segundos em milissegundos

num_intervalos = int(len(T) / intervalo_ms)

X_split =np.array_split(X, num_intervalos)
Y_split =np.array_split(Y, num_intervalos)
Z_split =np.array_split(Z, num_intervalos)
T_split =np.array_split(T, num_intervalos)


# Aplicar transformada de Fourier
transformadaZ = np.fft.fft(Z)
transformadaX = np.fft.fft(X)
transformadaY = np.fft.fft(Y)
frequencias = np.fft.fftfreq(len(T), T[1] - T[0])
print("TEMPO ", (T[len(T) -1]-T[0]))
# Plotar gráfico da frequência em função do tempo
plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(frequencias, np.abs(transformadaZ))
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro de Frequência - Eixo Z')
plt.grid(True)

plt.subplot(312)
plt.plot(frequencias, np.abs(transformadaY))
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro de Frequência - Eixo Y')
plt.grid(True)

plt.subplot(313)
plt.plot(frequencias, np.abs(transformadaX))
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Espectro de Frequência - Eixo X')
plt.grid(True)
plt.show()
