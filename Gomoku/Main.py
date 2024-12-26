# User inference version 0.0.1: 2024/12/21
# User inference version 0.0.2: 2024/12/
# No GUI, just a simple game that can be played with terminal
import numpy as np
import Logics as lg
test = lg.game(lim= 0)
state = 0
while not (state == -2):
    x = int(input())
    y = int(input())
    state = test.play(x, y)
    print(state)
    print(test.board)
exit(0)
