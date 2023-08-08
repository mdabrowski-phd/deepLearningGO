#%% Listing 3.29 Setting up a script so you can play your own bot
from dlgo.agent import naive
from dlgo import goboard_fast as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords
from six.moves import input
import time

BOARD_SIZE = 5

def main():
    game = goboard.GameState.new_game(BOARD_SIZE)
    bot = naive.RandomBot()
    
    while not game.is_over():

        print(chr(27) + "[2J")
        print_board(game.board)
        if game.next_player == gotypes.Player.black:
            human_move = input('-- ')
            point = point_from_coords(human_move.strip())
            move = goboard.Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        
        time.sleep(1.5)
        game = game.apply_move(move)

if __name__ == '__main__':
    main()