#%%
# pip install -r requirements.txt
print('[INFO] Carregando...')
print('[INFO] Importando módulos...')

from modules import Comandos_Respostas
comandos = Comandos_Respostas.comandos
respostas = Comandos_Respostas.respostas

from modules import porta
conexao_porta = porta.abrir_conexao_porta()

# Sintetizador de voz
import pyttsx3

# Reconhecimento da voz
import speech_recognition as sr

# Para executar arquivos de áudio
from playsound import playsound

# Obter informaçãoes de data e hora atuais e locais
import locale
locale.setlocale(locale.LC_ALL, 'pt_BR.utf8')
import random
import datetime
from datetime import timedelta
date = datetime.date.today().strftime("%d/%B/%Y")
date = date.split('/')

# Para inferir pesquisar ao navegador
import webbrowser as wb

# Para trabalhar com os modelos de classificação
import seaborn as sns
sns.set()#%%

#%%
respostas[5]
#%%
# Nome do(a) assistente
meu_nome = 'Ana'

# Seu próprio nome
seu_nome = 'Dalton'

# Path até o executável do Google Chrome
chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'
#%%
# Função para executar uma pesquisa no Google
def search(frase):
    wb.get(chrome_path).open('https://www.google.com/search?q=' + frase)

# Sintetizador de voz, para que o assistente possa responder
def speak(audio):
    engine = pyttsx3.init()  # object creation

    """ RATE"""
    rate = engine.getProperty(
        'rate')
    engine.setProperty('rate', 200)

    """VOLUME"""
    volume = engine.getProperty(
        'volume')

    engine.setProperty('volume', 1)    

    """VOICE"""
    voices = engine.getProperty('voices') 

    engine.setProperty('voice', voices[2].id)
    
    engine.say(audio)

    """Saving Voice to a file"""
    # engine.save_to_file('Hello World', 'test.mp3')

    engine.runAndWait()

# Função para ouvir e reconhecer a fala
def listen_microphone():
    # Habilita o microfone do usuário
    microfone = sr.Recognizer()

    # usando o microfone
    with sr.Microphone() as source:

        # Chama um algoritmo de reducao de ruidos no som
        microfone.adjust_for_ambient_noise(source, duration=0.8)

        # Frase para o usuario dizer algo
        print("Ouvindo: ")

        # Armazena o que foi dito numa variavel
        audio = microfone.listen(source)

        # Gravar o áudio falado
        with open('recordings/speech.wav', 'wb') as f:
            f.write(audio.get_wav_data())

    try:
        # Passa a variável para o algoritmo reconhecedor de padroes
        frase = microfone.recognize_google(audio, language='pt-BR')

        # Retorna a frase pronunciada
        print("Você disse: " + frase)

    # Se não reconheceu o padrão de fala, exibe a mensagem
    except sr.UnknownValueError:
        frase = ''
        print("Não entendi")

    return frase

# Controlador de excução da música
playing = False
count = 0

print('[INFO] Pronto para começar!')
#%%
while (1):       
    result = listen_microphone()    
       
    # Verificar se o nome da assistente foi falado corretamente, se ele está presente no comando e depois removê-lo do comando
    if meu_nome in result:
        result = str(result.split(meu_nome + ' ')[1])
        result = result.lower()

        if result in comandos[0]:
            playsound('n2.mp3')
            speak('Até agora minhas funções incluem: ' + respostas[0])

        if result in comandos[4]:
            playsound('n2.mp3')
            speak('Agora são ' + datetime.datetime.now().strftime('%H:%M'))

        if result in comandos[5]:
            playsound('n2.mp3')
            speak('Hoje é dia' + date[0] + ' de ' + date[1])

        if result in comandos[1]:
            playsound('n2.mp3')
            result = listen_microphone()
            anotacao = open("anotacao.txt", mode="a+", encoding="utf-8")
            anotacao.write(result + '\n')
            anotacao.close()
            speak(''.join(random.sample(respostas[1], k=1)))

        if result in comandos[2]:
            playsound('n2.mp3')
            speak(''.join(random.sample(respostas[2], k=1)))
            result = listen_microphone()
            search(result)

        if result in comandos[7]:
            playsound('n2.mp3')
            conexao_porta.write(b'1')
            speak(''.join(random.sample(respostas[5], k=1)))
            
        if result == 'encerrar':
            playsound('n2.mp3')
            speak(''.join(random.sample(respostas[4], k=1)))
            break
    
    else:
        playsound('n3.mp3')
# %%
