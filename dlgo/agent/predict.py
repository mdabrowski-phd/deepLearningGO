#%% Listing 8.1 Initializing an agent with a Keras model and a Go board encoder
import numpy as np

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo.encoders.base import get_encoder_by_name

from dlgo import goboard
from dlgo import kerasutil

class DeepLearningAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder

#%% Listing 8.2 Encoding board state and predicting move probabilities with a model
    def predict(self, game_state):
        encoded_state = self.encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        return self.model.predict(input_tensor)[0]

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state)

#%% Listing 8.3 Scalling, clipping, and renormalizing your move probability distribution
        move_probs = move_probs ** 3
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps) # prevents move prob. from getting stuck at 0 or 1
        move_probs = move_probs / np.sum(move_probs)

#%% Lisitng 8.4 Trying to apply moves from a ranked candidate list
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                    not is_point_an_eye(game_state.board, point, game_state.next_player):
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()

#%% Listing 8.5 Serializing a deep-learning agent
    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])

#%% Listing 8.6 Deserializing a DeepLearningAgent from an HDF5 file
def load_prediction_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = get_encoder_by_name(
        encoder_name, (board_width, board_height))
    return DeepLearningAgent(model, encoder)