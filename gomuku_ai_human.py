#gomoku env build based on Kairan Cao(my friend)'s gomoku game, 
import time
import sys
import numpy as np
import pygame
import random
from stable_baselines3 import PPO
from gomoku_env import GomokuEnv
from CustomPolicy import CustomCnnPolicy

WOOD = (0xd4, 0xb8, 0x96)
BLACK = (0, 0, 0)
WHITE = (0xff, 0xff, 0xff)
RED = (0xff, 0, 0)

lines = []  # a list of all the lines on the board


# coordinate of all the horizontal lines
for i in range(19):
    lines.append((np.ones(19, dtype=np.int8) * i, np.arange(19, dtype=np.int8)))

# vertical
for i in range(19):
    lines.append((np.arange(19, dtype=np.int8), np.ones(19, dtype=np.int8) * i))

# top left to bottom right
for i in range(4, 19):
    lines.append((np.arange(i + 1, dtype=np.int8), np.arange(i + 1, dtype=np.int8) + 18 - i))
for i in range(4, 18):
    lines.append((np.arange(i + 1, dtype=np.int8) + 18 - i, np.arange(i + 1, dtype=np.int8)))

# the other diagonal
for i in range(4, 19):
    lines.append((-np.arange(i + 1, dtype=np.int8) + 18, np.arange(i + 1, dtype=np.int8) + 18 - i))
for i in range(4, 18):
    lines.append((-np.arange(i + 1, dtype=np.int8) + i, np.arange(i + 1, dtype=np.int8)))


class Board:
    def __init__(self, player=1):
        """Creates a Board object, representing a Gomoku chessboard"""
        self.finished = False
        self.player = player
        self.board = np.zeros((19, 19), dtype=np.int8)
        self.moves_count = 0
        self.played_pos = list()
        self.last_length = [0, 0]
        self.contiunous_num = np.zeros((2, 3), dtype=np.uint8)

    def play(self, pos):
        """Play on the given position"""
        pos = tuple(pos)

        self.board[pos] = self.player  # update the board

        self.played_pos.append(pos)
        
        if self.is_finished(pos):
            self.finished = True
            return
        self.player = -1 * self.player

    def is_finished(self, played_pos):
        """Check if the game is finished, returns 1 if the p2 wins, -1 if p1, and 0 if game is not finished"""
        if self.is_won(played_pos):
            return self.board[tuple(played_pos)]
        return 0

    def has_pos(self, player, positions):
        """Returns an array of 1 dimension, size of the length positions, of if the positions are occupied by the p1"""
        has_pos_bool = np.empty(len(positions), dtype=np.bool)
        index = 0
        for position in positions:
            has_pos_bool[index] = (self.board[tuple(position)] == player)
            index += 1
        return has_pos_bool


    def is_won(self, played_pos, player=None):
        """Returns true if the game is won by the given move (can be played or not yet played), else returns None"""
        # checks for vertical, horizontal, top-left-to-bottom-right, and the other diagonal lines, for dim = 0, 1, None, None respectively (with special logic for the last case)
        if player is None:
            player = self.player
        played_pos = tuple(played_pos)
        original_stone = self.board[played_pos]
        self.board[played_pos] = player
        top_left_to_p2_right_is_checked = False
        for dim in (0, 1, None, None):
            positions = list()  # a list of positions in a line, will later check if there are 5 consecutive positions in the list are of the same color
            for offset in range(-5, 6):
                pos_in_line = np.array(played_pos, dtype=np.int8)

                if top_left_to_p2_right_is_checked:
                    # goes from top right to bottom left
                    pos_in_line[0] -= offset
                    pos_in_line[1] += offset
                else:
                    pos_in_line[dim] += offset

                positions.append(pos_in_line)

            if dim is None:  # distinguish the 2 diagonals
                top_left_to_p2_right_is_checked = True

            positions = np.unique(np.clip(np.array(positions, dtype=np.int8), 0, 18), axis=0).tolist()
            occupied = self.has_pos(player, positions).tolist()  # 1d array of boolean values of if the positions on the line is occupied
            # add False at the start and end of "occupied" to ensure that it works properly at the edge of board
            occupied.insert(False, 0)
            occupied.append(False)

            occupied = np.array(occupied, dtype=np.bool)
            dist = np.diff(np.where(occupied == False))  # the distance between the positions not occupied by p1
            
            self.last_length[0 if player == 1 else 1] = dist.max() - 1
            
            if dist.max() == 3:  # if the p1 occupied exactly 2 in a row
                self.contiunous_num[0 if player == 1 else 1, 0] += 1
            
            if dist.max() == 4:  # if the p1 occupied exactly 3 in a row
                self.contiunous_num[0 if player == 1 else 1, 1] += 1
                
            if dist.max() == 5:  # if the p1 occupied exactly 4 in a row
                self.contiunous_num[0 if player == 1 else 1, 2] += 1
            
            if dist.max() > 5:  # if the p1 occupied 5 or more in a row
                self.player = player * -1
                self.board[played_pos] = original_stone
                return True
                
        self.board[played_pos] = original_stone

def highlight(screen,game_pos, color=RED):
    """highlights a stone"""

    pos_x,pos_y = np.flip(game_pos) * 40 + 5   #find the top-left corner of the square the stone is in, and shift slightly
    vertices = (pos_x,pos_y),(pos_x+5,pos_y),(pos_x,pos_y+5)
    pygame.draw.polygon(screen,color,vertices)



def main():
    board = Board()

    policy_kwargs = dict(features_extractor_class=CustomCnnPolicy, features_extractor_kwargs=dict(features_dim=512),)

    model = PPO.load("best_ppo_models/best_model/.", verbose=1, policy_kwargs=policy_kwargs)
    print("model loaded")
    
    pygame.init()

    # Set the width and height of the screen [width, height]
    size = (760, 760)
    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("Gomoku Game")
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    screen.fill(WOOD)  # Draw the background color of the board
    for x in range(20, 760, 40):  # Draw the vertical lines on the board
        pygame.draw.line(screen, BLACK, (x, 20), (x, 740))
    for y in range(20, 760, 40):  # Draw the horizontal lines on the board
        pygame.draw.line(screen, BLACK, (20, y), (740, y))
    pygame.display.flip()
    
    last_eight_moves = np.full((2, 4, 2), 255, dtype=np.uint8)
    
    # -------- Main Program Loop -----------
    while not board.finished:
        # --- Main event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                board.finished = True

            elif event.type == pygame.MOUSEBUTTONDOWN:  # Mouse click

                if event.button == 1:  # Left click
                    mouse_pos = np.array(pygame.mouse.get_pos())
                    game_pos = np.flip(mouse_pos // 40)  # translate the mouse position into a position on the board
                    if board.board[tuple(game_pos)] == 0:  # if this play is valid
                        pygame.draw.circle(screen, BLACK, np.flip(game_pos) * 40 + 20, 15)
                        try:
                            highlight(screen, board.played_pos[-1], WOOD) # remove the highlighting of the previous p2 move
                        except IndexError:
                            pass
                        highlight(screen, game_pos)
                        pygame.display.flip()
                        np.append(last_eight_moves[0], [game_pos[0], game_pos[1]])
                        np.delete(last_eight_moves[0], 0)
                        board.play(game_pos)
            
        if board.player == -1:
            new_board = board.board.copy() * board.player
        
            player = (new_board == 1).astype(np.uint8) * 255# current player is always 1, as black
            opponent = (new_board == -1).astype(np.uint8) * 255
            layers = [player, opponent]
        
            for user in last_eight_moves:# add last 8 moves
                for move in user:
                    layer = np.zeros((19, 19), dtype=np.uint8)
                    if move[0] != 255 or move[1] != 255:
                        layer[move[0], move[1]] = 255
                    layers.append(layer)

            obs = np.stack(layers, axis=-1).astype(np.uint8)
            
            # print(obs)
            
            model_action, _states = model.predict(obs, deterministic=True)
            x, y = divmod(model_action, 19)
            while board.board[x, y] != 0:
                # print("AI attempted illegal move at position:", (x, y))
                model_action, _states = model.predict(obs, deterministic=False)
                x, y = divmod(model_action, 19)
            pygame.draw.circle(screen, WHITE, np.flip((x, y)) * 40 + 20, 15)
            try:
                highlight(screen, board.played_pos[-1], WOOD) # remove the highlighting of the previous p2 move
            except IndexError:
                pass
            highlight(screen, (x, y))
            pygame.display.flip()
            np.append(last_eight_moves[1], [x, y])
            np.delete(last_eight_moves[1], 0)
            board.play((x, y))
            print("AI played at position:", (x, y))
                        
    print("Game over! The winner is player", "2 (White)" if board.player == -1 else "1 (Black)")
    # Close the window and quit.
    time.sleep(1)
    pygame.quit()

if __name__ == "__main__":
    main()