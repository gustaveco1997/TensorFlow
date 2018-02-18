import pygame
from gtts import gTTS  # importamos o modúlo gTTS

pygame.mixer.pre_init(27000, -16, 2, 4096) # setup mixer to avoid sound lag
pygame.init()


class Voz:
    def __init__(self):
        pass

    def entrada(self, frase):
        # TRANSFORMANDO TEXTO EM FALA - USANDO API DO GOOGLE
        voz = gTTS(frase, lang="pt")  # guardamos o nosso texto na variavel voz
        voz.save("Classes/voz.mp3")  # salvamos com o comando save em mp3
        self.reproduzir()

    def reproduzir(self):

        pygame.mixer.music.load("Classes/voz.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pass
