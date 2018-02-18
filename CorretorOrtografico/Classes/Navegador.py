import webbrowser as navegador


def pesquisarGoogle(termo):
        navegador.open_new("https://www.google.com.br/search?q=" + termo)

def abrirSite(termo):
        navegador.open_new("http://" + termo)
