import time
import sys
import numpy as np
import pygame
import gymnasium.spaces as spaces
from gomoku import Board, highlight
from gymnasium import Env
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from CustomPolicy import CustomExtractor, CustomActorCriticPolicy
from torch import optim
import sys

WOOD = (0xd4, 0xb8, 0x96)
BLACK = (0, 0, 0)
WHITE = (0xff, 0xff, 0xff)
RED = (0xff, 0, 0)

class GomokuEnv(Env):
    def __init__(self, render = False, wait_time = 1, set_up=False):
        self.action_space = spaces.Discrete(19*19)
        self.observation_space = spaces.Box(low=0, high=255, shape=(19, 19, 10), dtype=np.uint8)
        self.board = Board()
        self.n_step = 0
        self.last_eight_moves = np.full((2, 4, 2), 255, dtype=np.uint8)# record last 4 moves of both players, initialized to invalid positions
        self.render = render
        self.wait_time = wait_time
        
        if render:
            # display
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
        
        policy_kwargs = dict(features_extractor_class=CustomExtractor, features_extractor_kwargs=dict(features_dim=512), optimizer_class=optim.AdamW, optimizer_kwargs=dict(weight_decay=1e-5))
        if not set_up:
            self.model =  MaskablePPO.load("./best_ppo_models/best_model", verbose=1, policy_kwargs=policy_kwargs)
    
    def draw(self, x, y, color):
        pygame.event.pump()
        pygame.draw.circle(self.screen, color, np.flip((x, y)) * 40 + 20, 15)
        try:
            highlight(self.screen, self.board.played_pos[-1], WOOD) # remove the highlighting of the previous player's move
        except IndexError:
            pass
        highlight(self.screen, (x, y))
        pygame.display.flip()
        time.sleep(self.wait_time)
    
    def step(self, action):
        
        x, y = divmod(action, 19)
        
        if self.render:
            self.draw(x, y, BLACK)
        
        self.last_eight_moves[0] = np.append(self.last_eight_moves[0][1:], [[x, y]], axis=0)
        self.board.play((x, y))# record the move of current player
        sys.stdout.write("Player 1 (Black) played at position" + str((x, y)) + "\n")
        
        self.n_step += 1
        info = {"n_steps": self.n_step, "action_mask": self.legal_moves}
        
        if self.board.finished:# game over
            reward = 10
            sys.stdout.write("Game over, the winner is 1 (Black) in " + str(self.n_step) + " steps" + "\n")
            return self.observation, reward, True, False, info
        elif self.n_step >= 19*19 - 1:
            reward = -5
            sys.stdout.write("Game over, draw in " + str(self.n_step) + " steps" + "\n")
            return self.observation, reward, True, False, info
        
        model_action, _states = self.model.predict(self.observation, deterministic=True, action_masks=self.legal_moves)
        x, y = divmod(model_action, 19)
        
        if self.render:
            self.draw(x, y, WHITE)
        
        self.last_eight_moves[1] = np.append(self.last_eight_moves[1][1:], [[x, y]], axis=0)
        self.board.play((x, y))# record the move of current player
        sys.stdout.write("Player 2 (White) played at position" + str((x, y)) + "\n")
        
        self.n_step += 1
        info = {"n_steps": self.n_step, "action_mask": self.legal_moves}
        
        if self.board.finished:# game over
            reward = -10
            sys.stdout.write("Game over, the winner is 2 (White) in " + str(self.n_step) + " steps" + "\n")
            return self.observation, reward, True, False, info
        elif self.n_step >= 19*19 - 1:
            reward = -5
            sys.stdout.write("Game over, draw in " + str(self.n_step) + " steps" + "\n")
            return self.observation, reward, True, False, info

        sys.stdout.flush()
        
        return self.observation, 0, False, False, info
    
    def reset(self, seed=None, options=None):
        if self.render:
            pygame.event.pump()
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
        self.n_step = 0
        self.last_eight_moves = np.full((2, 4, 2), 255, dtype=np.uint8)
        policy_kwargs = dict(features_extractor_class=CustomExtractor, features_extractor_kwargs=dict(features_dim=512), optimizer_class=optim.AdamW, optimizer_kwargs=dict(weight_decay=1e-5))
        self.model =  MaskablePPO.load("./best_ppo_models/best_model", verbose=1, policy_kwargs=policy_kwargs)
        info = {"n_steps": self.n_step, "action_mask": self.legal_moves}
        
        return self.observation, info
    
    def close(self):
        if self.render:
            pygame.quit()
    
    @property
    def observation(self):
        new_board = self.board.board.copy() * self.board.player
        
        player = (new_board == 1).astype(np.uint8) * 255# current player is always 1, as black
        opponent = (new_board == -1).astype(np.uint8) * 255
        layers = [player, opponent]
        
        for user in self.last_eight_moves:# add last 8 moves
            for move in user:
                layer = np.zeros((19, 19), dtype=np.uint8)
                if move[0] != 255 or move[1] != 255:
                    layer[move[0], move[1]] = 255
                layers.append(layer)
        
        return np.stack(layers, axis=-1).astype(np.uint8)
    
    @property
    def legal_moves(self):
        # Flattened mask: 1 = legal, 0 = illegal
        return (self.board.board.flatten() == 0).astype(np.int8)