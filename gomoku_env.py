import time
import sys
import numpy as np
import pygame
import random
from gomoku import Board, highlight
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

WOOD = (0xd4, 0xb8, 0x96)
BLACK = (0, 0, 0)
WHITE = (0xff, 0xff, 0xff)
RED = (0xff, 0, 0)

render = False
endgame_if_illegal = False
enable_illegal_move_notification = False
enable_illegal_move_penalty = False

class GomokuEnv(Env):
    def __init__(self):
        self.action_space = Discrete(19*19)
        self.observation_space = Box(low=0, high=255, shape=(19, 19, 10), dtype=np.uint8)#10 layers: 2 for player and opponent, 4 for last 8 moves of player, 4 for last 8 moves of opponent
        self.board = Board()
        self.total_reward = 0.0
        self.wait_time = 0.2  # time to wait after each move(s)
        self.n_step = 0
        self.last_eight_moves = np.full((2, 4, 2), 255, dtype=np.uint8)# record last 8 moves of both players, initialized to invalid positions
        
        if render:
            #display
            pygame.init()
            # Set the width and height of the screen [width, height]
            self.size = (760, 760)
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("Gomoku Game")
            # Used to manage how fast the screen updates
            self.clock = pygame.time.Clock()
            self.screen.fill(WOOD)  # Draw the background color of the board
            for x in range(20, 760, 40):  # Draw the vertical lines on the board
                pygame.draw.line(self.screen, BLACK, (x, 20), (x, 740))
            for y in range(20, 760, 40):  # Draw the horizontal lines on the board
                pygame.draw.line(self.screen, BLACK, (20, y), (740, y))
            pygame.display.flip()
        
    def step(self, action):
        x, y = divmod(action, 19)
        if self.board.board[x, y] != 0:# illegal move filter
            info = {"illegal_move": True, "total_reward": self.total_reward, "reward": -100, "n_step": self.n_step}
            if enable_illegal_move_notification:
                sys.stdout.write("Illegal move at position " + str((x, y)) + "\n")
                sys.stdout.flush()
            ill_reward = 0
            if enable_illegal_move_penalty:
                self.total_reward -= 1000
                ill_reward = -1000
            return self.observation, ill_reward, False, endgame_if_illegal, info
        
        if render:
            pygame.event.pump()
            if self.board.player == 1:
                pygame.draw.circle(self.screen, BLACK, np.flip((x, y)) * 40 + 20, 15)
            else:
                pygame.draw.circle(self.screen, WHITE, np.flip((x, y)) * 40 + 20, 15)
            try:
                highlight(self.screen, self.board.played_pos[-1], WOOD) # remove the highlighting of the previous player's move
            except IndexError:
                pass
            highlight(self.screen, (x, y))
            pygame.display.flip()
            time.sleep(self.wait_time)
        
        # record the move of current player
        np.append(self.last_eight_moves[0 if self.board.player == 1 else 1], [x, y])
        np.delete(self.last_eight_moves[0 if self.board.player == 1 else 1], 0)
        
        self.board.play((x, y))
        
        terminated = self.board.finished or self.n_step >= 19*19
        if self.board.finished:# game over
            sys.stdout.write("Game over, the winner is: " + ("2 (White)" if self.board.player == -1 else "1 (Black)" + "\n"))
            sys.stdout.flush()

        self.total_reward += self.reward
        self.n_step += 1
        info = {"invalid_move": False, "total_reward": self.total_reward, "reward": self.reward, "n_step": self.n_step}
        
        return self.observation, self.reward, terminated, False, info
    
    def reset(self, seed=None, options=None):
        sys.stdout.write("reward: " + str(self.total_reward) + " in " + str(self.n_step) + " steps" + "\n")
        sys.stdout.flush()
        
        if render:
            #display
            pygame.quit()
            pygame.init()
            # Set the width and height of the screen [width, height]
            self.size = (760, 760)
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("Gomoku Game")
            # Used to manage how fast the screen updates
            self.clock = pygame.time.Clock()
            self.screen.fill(WOOD)  # Draw the background color of the board
            for x in range(20, 760, 40):  # Draw the vertical lines on the board
                pygame.draw.line(self.screen, BLACK, (x, 20), (x, 740))
            for y in range(20, 760, 40):  # Draw the horizontal lines on the board
                pygame.draw.line(self.screen, BLACK, (20, y), (740, y))
            pygame.display.flip()
        
        self.board = Board()
        self.total_reward = 0.0
        self.n_step = 0
        self.max_length = [0, 0]
        info = {"invalid_move": False, "total_reward": self.total_reward, "reward": self.reward, "n_step": self.n_step}
        
        return self.observation, info
    
    def close(self):
        if render:
            pygame.quit()
    
    @property
    def observation(self):
        new_board = self.board.board.copy() * self.board.player
        
        player = (new_board == 1).astype(np.uint8)# current player is always 1, as black
        opponent = (new_board == -1).astype(np.uint8)
        layers = [player, opponent]
        
        for user in self.last_eight_moves:# add last 8 moves
            for move in user:
                layer = np.zeros((19, 19), dtype=np.uint8)
                if move[0] != 255 or move[1] != 255:
                    layer[move[0], move[1]] = 1
                layers.append(layer)
        
        return np.stack(layers, axis=-1).astype(np.uint8)
    
    @property
    def reward(self):
        self_reward = 0.0
        opp_reward = 0.0
        constant_reward = -0.2  # constant reward for each valid move to encourage longer games
        
        # reward shaping based on continuous pieces
        if self.board.last_length[0 if self.board.player == 0 else 1] >= 5:
            self_reward += 1000
        elif self.board.last_length[0 if self.board.player == 0 else 1] == 4:
            self_reward += 100.0
        elif self.board.last_length[0 if self.board.player == 0 else 1] == 3:
            self_reward += 1.0
        elif self.board.last_length[0 if self.board.player == 0 else 1] == 2:
            self_reward += 0.5
        
        # penalty for opponent's continuous pieces
        if self.board.last_length[1 if self.board.player == 0 else 0] >= 5:
            self_reward -= 500
        elif self.board.last_length[1 if self.board.player == 0 else 0] == 4:
            opp_reward -= 50.0
        elif self.board.last_length[1 if self.board.player == 0 else 0] == 3:
            opp_reward -= 1.0
            
        return self_reward + opp_reward + constant_reward