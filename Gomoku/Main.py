# User inference version 0.0.1: 2024/12/21
# User inference version 0.0.2: 2024/12/27
# Simple GUI for 2 players
import numpy as np
import Logics as lg
import pygame

test = lg.game(19,lim= 0)

pygame.init()
WOOD = (0xd4,0xb8,0x96)
BLACK = (0,0,0)
WHITE = (255,255,255)
size = (760, 760)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Gomoku Game")

clock = pygame.time.Clock()

screen.fill(WOOD)
for x in range(20, 760, 40):
    pygame.draw.line(screen, BLACK, (x, 20), (x, 740))
for y in range(20, 760, 40):
    pygame.draw.line(screen, BLACK, (20, y), (740, y))
pygame.display.flip()

state = 0
done = False
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            state = -2
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if done:
                    pygame.quit()
                    quit()
                mouse_pos = np.array(pygame.mouse.get_pos())
                game_pos = np.flip(mouse_pos // 40)
                state = test.play(game_pos[0],game_pos[1])
                if state == 0:
                    color = BLACK if test.color == 1 else WHITE 
                    pygame.draw.circle(screen,color,np.flip(game_pos)*40+20,15)
                    pygame.display.flip()
                elif state == -2:
                    color = BLACK if test.color != 1 else WHITE 
                    pygame.draw.circle(screen,color,np.flip(game_pos)*40+20,15)
                    pygame.display.flip()
                    done = True
exit(0)
