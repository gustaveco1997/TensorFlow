import spotipy

class Spotify:
    def __init__(self):
        sp = spotipy.Spotify()

        results = sp.search(q='Gustavo Reis', limit=20)
        for i, t in enumerate(results['tracks']['items']):
            print(' ', i, t['name'])

a = Spotify()