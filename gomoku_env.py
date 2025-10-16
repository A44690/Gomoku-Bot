import time
import sys
import numpy as np
import pygame
import torch as th
import os
import gymnasium.spaces as spaces
from gomoku import Board, highlight
from gymnasium import Env
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from CustomPolicy import CustomExtractor, CustomActorCriticPolicy
from torch import optim

WOOD = (0xd4, 0xb8, 0x96)
BLACK = (0, 0, 0)
WHITE = (0xff, 0xff, 0xff)
RED = (0xff, 0, 0)

class GomokuEnv(Env):
    def __init__(self, render = False, wait_time = 1.0, eval_mode = False, models_folder_path = "ppo_models/", best_model_opponent_percentage = 0.25):
        self.action_space = spaces.Discrete(19*19)
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(10, 19, 19), 
            dtype=np.uint8
            )
        self.board = Board()
        self.n_step = 0
        self.last_eight_moves = np.full((2, 4, 2), 255, dtype=np.uint8)# record last 4 moves of both players, initialized to invalid positions
        self.render = render
        self.wait_time = wait_time
        self.random_move = False
        self.color = BLACK
        self.eval_mode = eval_mode
        self.second_start_overide = True
        self.step_count = 0
        self.policy_kwargs = dict(
            features_extractor_class=CustomExtractor, 
            features_extractor_kwargs=dict(features_dim=2*19*19 + 19*19), 
            optimizer_class=optim.AdamW, 
            optimizer_kwargs=dict(weight_decay=1e-5)
            )
        self.model_list = []
        self.best_model_opponent_percentage = best_model_opponent_percentage
        self.last_model_index = -2
        
        for file in os.listdir(models_folder_path):
            if file.endswith(".zip"):
                self.model_list.append(models_folder_path + file)
        
        self.load_opponent_model()
        
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
    
    def load_opponent_model(self):
        #select random opponent model from the models folder
        random_number = np.random.rand()
        
        if random_number < self.best_model_opponent_percentage and self.last_model_index != -1: #play with best model
            sys.stdout.write("Opponent will play with the best model\n")
            self.last_model_index = -1
            try:
                self.model = MaskablePPO.load("best_ppo_models/best_model", verbose=1, policy_kwargs=self.policy_kwargs)
            except:
                sys.stdout.write("Error, opponent will play randomly\n")
                self.random_move = True
        else: #play with an older model or randomly
            random_number = int(random_number / (1 - self.best_model_opponent_percentage) * (len(self.model_list) + 1))
            if random_number != self.last_model_index: #avoid playing with the same model twice in a row
                if random_number < len(self.model_list): #play with an older model
                    model_path = self.model_list[random_number]
                    sys.stdout.write("Selected opponent model: " + model_path + "\n")
                    try:
                        self.model = MaskablePPO.load(model_path, verbose=1, policy_kwargs=self.policy_kwargs)
                    except:
                        sys.stdout.write("Error, opponent will play randomly\n")
                        self.random_move = True
                else: #play randomly
                    sys.stdout.write("Opponent will play randomly\n")
                    self.random_move = True
                self.last_model_index = random_number
    
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
    
    def opponent_move(self, color):
        
        x, y = -1, -1
        
        if self.random_move:
            legal_positions = np.argwhere(self.board.board == 0)
            rad = np.random.choice(len(legal_positions))
            x, y = legal_positions[rad]
        else:
            model_action, _states = self.model.predict(self.opponent_observation, deterministic=True, action_masks=self.legal_moves)
            x, y = divmod(model_action, 19)
        
        if self.render:
            self.draw(x, y, color)
        
        self.last_eight_moves[1] = np.append(self.last_eight_moves[1][1:], [[x, y]], axis=0)
        self.board.play((x, y))# record the move of current player
        sys.stdout.write("Player 2 played at position" + str((x, y)) + "\n")
        
    def step(self, action):
        
        self.step_count += 1
        if self.step_count % 256 == 1 and self.eval_mode == False: #reload the model every 256 steps which should be right after the evaluation
            self.step_count = self.step_count % 256
            sys.stdout.write("Reloading the opponent model...\n")
            self.load_opponent_model()
            sys.stdout.write("Opponent model reloaded\n")
            
        # player's turn
        x, y = divmod(action, 19)
        if self.render:
            self.draw(x, y, self.color)
        self.last_eight_moves[0] = np.append(self.last_eight_moves[0][1:], [[x, y]], axis=0)
        self.board.play((x, y))# record the move of current player
        sys.stdout.write("Player 1 played at position" + str((x, y)) + "\n")
        
        self.n_step += 1
        info = {"n_steps": self.n_step, "action_mask": self.legal_moves}
        
        if self.board.finished:# game over
            reward = 1
            sys.stdout.write("Game over, the winner is 1 in " + str(self.n_step) + " steps" + "\n")
            if self.render:
                time.sleep(self.wait_time*10)
                
            self.second_start_overide = not self.second_start_overide# switch the starting player in eval mode
                
            return self.observation, reward, True, False, info
        elif self.n_step >= 19*19 - 1:
            reward = 0
            sys.stdout.write("Game over, draw in " + str(self.n_step) + " steps" + "\n")
            if self.render:
                time.sleep(self.wait_time*10)
                
            self.second_start_overide = not self.second_start_overide# switch the starting player in eval mode
                
            return self.observation, reward, True, False, info
        
        # opponent's turn
        self.opponent_move(WHITE if self.color == BLACK else BLACK)
        
        self.n_step += 1
        info = {"n_steps": self.n_step, "action_mask": self.legal_moves}
        
        if self.board.finished:# game over
            reward = -1
            sys.stdout.write("Game over, the winner is 2 in " + str(self.n_step) + " steps" + "\n")
            if self.render:
                time.sleep(self.wait_time*10)
                
            self.second_start_overide = not self.second_start_overide# switch the starting player in eval mode
                
            return self.observation, reward, True, False, info
        elif self.n_step >= 19*19 - 1:
            reward = -5
            sys.stdout.write("Game over, draw in " + str(self.n_step) + " steps" + "\n")
            if self.render:
                time.sleep(self.wait_time*10)
                
            self.second_start_overide = not self.second_start_overide# switch the starting player in eval mode
                            
            return self.observation, reward, True, False, info
        
        sys.stdout.flush()
        
        return self.observation, 0, False, False, info
    
    def reset(self, seed=None, options=None):
        if self.render:
            pygame.event.pump()
            self.screen.fill(WOOD)  # Draw the background color of the board
            for x in range(20, 760, 40):  # Draw the vertical lines on the board
                pygame.draw.line(self.screen, BLACK, (x, 20), (x, 740))
            for y in range(20, 760, 40):  # Draw the horizontal lines on the board
                pygame.draw.line(self.screen, BLACK, (20, y), (740, y))
            pygame.display.flip()
        
        self.board = Board()
        self.n_step = 0
        self.last_eight_moves = np.full((2, 4, 2), 255, dtype=np.uint8)
        self.color = BLACK
        
        if self.eval_mode:
            sys.stdout.write("Evaluation mode\n")
            if self.second_start_overide: 
                try:    
                    self.model = MaskablePPO.load("best_ppo_models/best_model", verbose=1, policy_kwargs=self.policy_kwargs)
                except:
                    sys.stdout.write("Model not found, opponent will play randomly\n")
                    self.random_move = True
        else:
            self.load_opponent_model()
            
        np.random.seed(seed)
        if (self.eval_mode and self.second_start_overide) or (np.random.rand() > 0.5 and not self.eval_mode):# opponent plays first
            sys.stdout.write("Player 2 plays first\n")
            self.color = WHITE
            self.opponent_move(BLACK)
            self.n_step += 1
        else:
            sys.stdout.write("Player 1 plays first\n")
        
        info = {"n_steps": self.n_step, "action_mask": self.legal_moves}
        
        return self.observation, info
    
    def close(self):
        if self.render:
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
        
        return np.stack(layers, axis=0).astype(np.uint8)
    
    @property
    def opponent_observation(self):
        new_board = self.board.board.copy() * self.board.player
        
        player = (new_board == 1).astype(np.uint8)# current player is always 1, as black
        opponent = (new_board == -1).astype(np.uint8)
        layers = [player, opponent]
        
        for user in self.last_eight_moves[::-1]:# add last 8 moves
            for move in user:
                layer = np.zeros((19, 19), dtype=np.uint8)
                if move[0] != 255 or move[1] != 255:
                    layer[move[0], move[1]] = 1
                layers.append(layer)
        
        return np.stack(layers, axis=0).astype(np.uint8)
    
    @property
    def legal_moves(self):
        # Flattened mask: 1 = legal, 0 = illegal
        return (self.board.board.flatten() == 0).astype(np.int8)