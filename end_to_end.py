import h5py

from keras.models import Sequential
from keras.layers import Dense

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import large

#%% Listing 8.8 Loading features and labels from Go data with a procesor
go_board_rows, go_board_cols = 19, 19
nb_classes = go_board_rows * go_board_cols
encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
processor = GoDataProcessor(encoder=encoder.name())

X, y = processor.load_go_data(num_samples=100)

#%% Listing 8.9 Building and running a large Go move-predicting model with Adadelta
input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
model = Sequential()
network_layers = large.layers(input_shape)
for layer in network_layers:
    model.add(layer)
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(X, y, batch_size=128, epochs=20, verbose=1)

#%% Listing 8.10 Creating and persisting a DeepLearningAgent
deep_learning_bot = DeepLearningAgent(model, encoder)
deep_learning_bot.serialize("E:/MACHINE_LEARNING/DeepLearningGoGame/dlgo_MD/agents/betago.hdf5")

#%% Listing 8.11 Loading a bot back into memory and serving it in a web application
model_file = h5py.File("E:/MACHINE_LEARNING/DeepLearningGoGame/dlgo_MD/agents/betago.hdf5", "r")
bot_from_file = load_prediction_agent(model_file)

web_app = get_web_app({'predict': bot_from_file})
web_app.run() # http://127.0.0.1:5000/static/play_predict_19.html