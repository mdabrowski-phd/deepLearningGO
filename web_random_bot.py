#%% Listing 8.7 Registering a random agent and starting a web application with it
from dlgo.agent.naive import RandomBot
from dlgo.httpfrontend.server import get_web_app

random_agent = RandomBot()
web_app = get_web_app({'random': random_agent})
web_app.run() # http://127.0.0.1:5000/static/play_random_99.html