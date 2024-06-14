import yaml
from ServerClass import ServerClass


# - inizializzazione strutture dati --

with open('conf/configuration.yml') as conf_file:
    params = yaml.safe_load(conf_file)

# -- fine inizializzazione strutture dati --

server = ServerClass(params) #viene richiamato il costruttore della classe ServerClass
server.start() #richiamo del metodo di avvio della classe ServerClasspip