import time
import pywifi
import matplotlib.pyplot as plt
import numpy as np

# Função para escanear as redes Wi-Fi próximas
def scan_wifi():
    wifi = pywifi.PyWiFi()
    iface = wifi.interfaces()[0]  # Pode haver várias interfaces, escolha a primeira

    iface.scan()
    time.sleep(2)
    scan_results = iface.scan_results()
    
    # Agora, vamos criar uma lista de dicionários com o nome da rede (SSID) e a força do sinal (signal)
    network_info = [{'SSID': result.ssid, 'Signal': result.signal} for result in scan_results]
    
    return network_info

# Função para criar um mapa de calor com base nas forças dos sinais Wi-Fi
def create_heatmap(scan_results):
    x = []
    y = []
    signal_strength = []

    for result in scan_results:
        x.append(result['Signal'])
        y.append(0)  # Apenas uma dimensão no mapa de calor
        signal_strength.append(result['Signal'])

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(100, 1), range=[[-100, 0], [0, 1]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    return heatmap, extent

# Função para exibir o mapa de calor em tempo real
def display_heatmap(heatmap, extent):
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, aspect='auto', origin='lower', cmap='hot')
    plt.xlabel('Sinal Wi-Fi (dBm)')
    plt.ylabel('Tempo')
    plt.title('Mapa de Calor do Sinal Wi-Fi em Tempo Real')
    plt.colorbar(label='Contagem de Pacotes')
    plt.pause(1)

if __name__ == "__main__":
    while True:
        scan_results = scan_wifi()
        heatmap, extent = create_heatmap(scan_results)
        display_heatmap(heatmap, extent)
