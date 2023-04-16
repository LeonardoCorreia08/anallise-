import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Definir o caminho completo do arquivo TXT
diretorio = r'D:\Projetospy\projeto'
nome_arquivo = 'sinaisvitais003 100dias DV2 RAxxx1.txt'
caminho = os.path.join(diretorio, nome_arquivo)

# Inicializar as listas que vão armazenar os dados
horas = []
batimentos_cardiacos = []
pressoes_arteriais = []
temperaturas_corporais = []

# Abrir o arquivo TXT e ler as informações
with open(caminho, 'r', encoding='utf-8') as f:
    linhas = f.readlines()
    # Extrair as informações de cada linha do arquivo TXT
    for linha in linhas:
        dados = linha.split()
        horas.append(dados[0])
        batimentos_cardiacos.append(dados[1])
        pressoes_arteriais.append(dados[2])
        temperaturas_corporais.append(dados[3])

# Criar o dataframe com as informações coletadas
data = {'Hr': horas, 'Bpm': batimentos_cardiacos, 'PA': pressoes_arteriais, 'TC': temperaturas_corporais}
df = pd.DataFrame(data)

# Exportar o dataframe para arquivo CSV
df.to_csv('D:/Projetospy/projeto/dataset.csv', index=False)

# Definir o caminho do arquivo CSV
caminho = r'D:\Projetospy/projeto/dataset.csv'

# Ler o arquivo CSV e criar um dataframe
df = pd.read_csv(caminho)

def preparar_dados(df):
    # Verificar se os valores estÃ£o dentro dos limites permitidos
    for i in range(len(df)):
        if df.loc[i, 'Bpm'] < 0 or df.loc[i, 'Bpm'] > 100:
            # Substituir o valor pela media entre o valor antecessor e posterior
            df.loc[i, 'Bpm'] = (df.loc[i-1, 'Bpm'] + df.loc[i+1, 'Bpm'])/2
        if df.loc[i, 'PA'] < 0 or df.loc[i, 'PA'] > 20:
            # Substituir o valor pela media entre o valor antecessor e posterior
            df.loc[i, 'PA'] = (df.loc[i-1, 'PA'] + df.loc[i+1, 'PA'])/2
        if df.loc[i, 'TC'] < 0 or df.loc[i, 'TC'] > 40:
            # Substituir o valor pela media entre o valor antecessor e posterior
            df.loc[i, 'TC'] = (df.loc[i-1, 'TC'] + df.loc[i+1, 'TC'])/2
    return df


# Aplicar a funcao preparar_dados ao dataframe
df = preparar_dados(df)


# Gerar uma amostra aleatoria com 1000 linhas (480, 960, 1920, 3840, 7680 )
dfa = df.sample(n=960, random_state=42)

# Salvar o dataset de amostra em um arquivo CSV
dfa.to_csv("D:/Projetospy/projeto/dataset_amostra.csv", index=False)

# Converter as colunas para numericas
dfa['Bpm'] = pd.to_numeric(dfa['Bpm'], errors='coerce')
dfa['PA'] = pd.to_numeric(dfa['PA'], errors='coerce')
dfa['TC'] = pd.to_numeric(dfa['TC'], errors='coerce')

# Segmentar os dados em pacotes de 24 amostras
num_linhas = dfa.to_numpy().shape[0]
pacotes = dfa.to_numpy().reshape(-1, 160, 3)
pad_value = np.nan
pad = ((0, num_linhas * 24 - dfa.shape[0]), (0, 0))
pad = [(0, 0)] + [(p, p) for p in pad] + [(0, 0)]
pacotes = np.pad(pacotes, pad_width=((0,0),(0,0),(0,2)), mode='constant', constant_values=pad_value)


# Calcular a media do batimento cardiaco em cada pacote
media_bpm = np.nanmean(pacotes[:,:,0], axis=1)

def calcular_correlacao(dfa):
    # Calculo da correlacao entre batimentos cardiacos e pressao arterial
    corr = np.corrcoef(dfa['Bpm'], dfa['PA'])[0, 1]
    
    # Adicionar a coluna "correlacao" ao dataframe
    dfa['correlacao'] = corr
    
    # Verificar o critério de tomada de decisão
    media_bpm = np.nanmean(pacotes[:,:,0], axis=1)
    if media_bpm.mean() > 80 and data['PA'].mean() > 16:
        print('Alerta! Batimento cardíaco elevado e pressão arterial elevada')
    
    return dfa


# Exemplo de uso da funcao
dfa = calcular_correlacao(dfa)
print(dfa)

# Leitura do arquivo de dados
data = dfa

# Dividir em pacotes de 24 amostras para cada parametro
bc = [data["Bpm"][i:i+24] for i in range(0, len(data), 24)]
pa = [data["PA"][i:i+24] for i in range(0, len(data), 24)]
tc = [data["TC"][i:i+24] for i in range(0, len(data), 24)]

# Aplicar ferramentas estatasticas descritivas

for i in range(len(bc)):
    print("Estata­sticas descritivas para o pacote", i+1, "de Bpm:")
    print("Media:", np.mean(bc[i]))
    print("Desvio padrao:", np.std(bc[i]))
    print("Mi­nimo:", np.min(bc[i]))
    print("Maximo:", np.max(bc[i]))
    print("")

for i in range(len(pa)):
    print("Estati­sticas descritivas para o pacote", i+1, "de PA:")
    print("Media:", np.mean(pa[i]))
    print("Desvio padrao:", np.std(pa[i]))
    print("Mi­nimo:", np.min(pa[i]))
    print("Maximo:", np.max(pa[i]))
    print("")

for i in range(len(tc)):
    print("Estata­sticas descritivas para o pacote", i+1, "de TC:")
    print("Media:", np.mean(tc[i]))
    print("Desvio padrao:", np.std(tc[i]))
    print("Minimo:", np.min(tc[i]))
    print("Maximo:", np.max(tc[i]))
    print("")
    
# Carregar o dataset de amostra
dfa = pd.read_csv("D:/Projetospy/projeto/dataset_amostra.csv")

# Plotar um grafico de linhas com os valores de Bpm da amostra
plt.plot(dfa["Bpm"])
plt.title("Sinal Batimento Cardiaco")
plt.xlabel("Amostra")
plt.ylabel("Amplitude")
plt.show()

# Plotar um grafico de linhas com os valores de PA da amostra
plt.plot(dfa["PA"])
plt.title("Sinal PressÃ£o Arterial")
plt.xlabel("Amostra")
plt.ylabel("Amplitude")
plt.show()

# Plotar um grafico de linhas com os valores de TC da amostra
plt.plot(dfa["TC"])
plt.title("Sinal Temperatura")
plt.xlabel("Amostra")
plt.ylabel("Amplitude")
plt.show()