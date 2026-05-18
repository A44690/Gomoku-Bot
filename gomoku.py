#gomoku env build based on Kairan Cao(my friend)'s gomoku game, 
import time
import sys
import numpy as np
import pygame
import random
import local_constants as c

WOOD = (0xd4, 0xb8, 0x96)
BLACK = (0, 0, 0)
WHITE = (0xff, 0xff, 0xff)
RED = (0xff, 0, 0)

BOARD_SIZE = c.BOARD_SIZE
WINDOW_SIZE = 800 #better be a multiple of BOARD_SIZE
STANDARD_SPACING = WINDOW_SIZE / BOARD_SIZE
SIDE_SIZE = STANDARD_SPACING / 2

print("\nBOARD_SIZE:", BOARD_SIZE)
print("WINDOW_SIZE:", WINDOW_SIZE)
print("SIDE_SIZE:", SIDE_SIZE)
print("STANDARD_SPACING:", STANDARD_SPACING)
print()


class Board:
    def __init__(self, player=1):
        """Creates a Board object, representing a Gomoku chessboard"""
        self.finished = False
        self.player = player
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.moves_count = 0
        self.played_pos = list()

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
            return True
        return False

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
        # revision by Liuxuanhao Alex Zhao, more direct than the previous version, worst case 4*9 iterations, O(1) time complexity, the last version is too complex with many bugs
        if player is None:
            player = self.player
        played_pos = tuple(played_pos) # find the previously played position
        directions = [(1, 0), 
                      (0, 1), 
                      (1, 1), 
                      (1, -1)] # horizontal, vertical, diagonal down, diagonal up
        
        for direction in directions:
            count = 0
            for offset in range(-4, 5):
                x_position = played_pos[0] + offset * direction[0]
                y_position = played_pos[1] + offset * direction[1]
                
                if (x_position < 0 or 
                    x_position >= BOARD_SIZE or 
                    y_position < 0 or 
                    y_position >= BOARD_SIZE): # out of bounds check
                    count = 0
                    continue
                
                if self.board[x_position, y_position] == player: # if occupied by the player
                    count += 1
                    if count >= 5: # counter for 5 in a row
                        return True
                else:
                    count = 0
        return False

def highlight(screen,game_pos, color=RED):
    """highlights a stone"""

    pos_x, pos_y = np.flip(game_pos) * STANDARD_SPACING + SIDE_SIZE * 0.25   #find the top-left corner of the square the stone is in, and shift slightly
    vertices = (pos_x, pos_y), (pos_x + SIDE_SIZE * 0.25, pos_y), (pos_x, pos_y + SIDE_SIZE * 0.25)
    pygame.draw.polygon(screen, color, vertices)



def main():
    board = Board()

    pygame.init()

    # Set the width and height of the screen [width, height]
    size = (WINDOW_SIZE, WINDOW_SIZE)
    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("Gomoku Game")
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    screen.fill(WOOD)  # Draw the background color of the board
    x = SIDE_SIZE
    while (x < WINDOW_SIZE):
        pygame.draw.line(screen, BLACK, (x, SIDE_SIZE), (x, WINDOW_SIZE - SIDE_SIZE))
        x += STANDARD_SPACING
    y = SIDE_SIZE
    while (y < WINDOW_SIZE):
        pygame.draw.line(screen, BLACK, (SIDE_SIZE, y), (WINDOW_SIZE - SIDE_SIZE, y))
        y += STANDARD_SPACING
    pygame.display.flip()
    
    # -------- Main Program Loop -----------
    while not board.finished:
        # --- Main event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                board.finished = True

            elif event.type == pygame.MOUSEBUTTONDOWN:  # Mouse click

                if event.button == 1:  # Left click
                    mouse_pos = np.array(pygame.mouse.get_pos())
                    game_pos = np.clip(np.flip(mouse_pos // STANDARD_SPACING), 0, BOARD_SIZE - 1).astype(int)  # translate the mouse position into a position on the board
                    if board.board[tuple(game_pos)] == 0:  # if this play is valid
                        if board.player == 1:
                            pygame.draw.circle(screen, BLACK, np.flip(game_pos) * STANDARD_SPACING + SIDE_SIZE, STANDARD_SPACING * 0.375)
                        else:
                            pygame.draw.circle(screen, WHITE, np.flip(game_pos) * STANDARD_SPACING + SIDE_SIZE, STANDARD_SPACING * 0.375)
                        try:
                            highlight(screen, board.played_pos[-1], WOOD) # remove the highlighting of the previous move
                        except IndexError:
                            pass
                        highlight(screen, game_pos)
                        pygame.display.flip()
                        board.play(game_pos)
    print("Game over! The winner is player", "2 (White)" if board.player == -1 else "1 (Black)")
    # Close the window and quit.
    time.sleep(1)
    pygame.quit()

if __name__ == "__main__":
    main()