# Sintetizador de voz
import pyttsx3

import locale

locale.setlocale(locale.LC_ALL, "pt_BR.utf8")


# Sintetizador de voz, para que o assistente possa responder
def speak(audio):
    engine = pyttsx3.init()  # object creation

    """ RATE"""
    rate = engine.getProperty("rate")
    engine.setProperty("rate", 150)

    """VOLUME"""
    volume = engine.getProperty("volume")

    engine.setProperty("volume", 1)

    """VOICE"""
    voices = engine.getProperty("voices")

    engine.setProperty("voice", voices[2].id)

    engine.say(audio)

    """Saving Voice to a file"""
    # engine.save_to_file('Hello World', 'test.mp3')

    engine.runAndWait()
