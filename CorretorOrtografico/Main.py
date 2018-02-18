from Classes.Voz import Voz
from Bot import Bot
from Constantes import FuncoesConst as fc
from Classes import Navegador

CODIGO_FUNCAO = "--func"

class Main:
    def __init__(self):
        bot = Bot(True)
        voz = Voz()
        retorno = ""

        while True:
            ultimoTermo = retorno
            nome = input("Digite:")
            retorno = bot.response(nome)

            if retorno == None or retorno == "":
                retorno = "Falha no sistema"

            if retorno.__contains__("--func"):
                self.tratarFuncao(retorno)
            else:
                voz.entrada(retorno)
                print(retorno)

    def tratarFuncao(self, funcao):
        funcao = funcao.replace(CODIGO_FUNCAO,"")

        if funcao == fc.ABRIR_NAVEGADOR:
            Navegador.abrirSite("")
        elif funcao == fc.PESQUISAR_GOOGLE:
            Navegador.pesquisarGoogle("")






Main()
