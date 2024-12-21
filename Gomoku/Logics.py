import numpy as np
class game:
    def __init__(self, dimentions = 15, lim = 2**63 - 1):
        self.round = 0
        self.round_lim = lim
        self.dimention = dimentions
        self.board = np.zeros((dimentions, dimentions), dtype= np.uint8)
        self.color = 1
        self.count = 0
        self.check_dir = [[[1, 0], [-1, 0]],
                          [[0, 1], [0, -1]],
                          [[1, 1], [-1, -1]],
                          [[1, -1], [-1, 1]]]

    def play(self, x = 0, y = 0):
        if not (self.board[x][y] == 0) or not (x >= 0 and x < self.dimention) or not (y >= 0 and y < self.dimention):
            print("Error: Attempt to place a stone on a location where a stone already exist")
            return -1
        elif self.color == 1:
            self.board[x][y] = 1
        else:
            self.board[x][y] = 2
        self.round += 1
        if self.round > self.round_lim and not (self.round_lim == 0):
            print("Ended: The game had reached its final round")
            return -2
        win = False
        for i in range(4):
            self.count = 0
            if self.check(x, y, x, y, i):
                print("Ended: The player with color", str(self.color), "won the game")
                return -2
        if self.color == 1:
            self.color = 2
        else:
            self.color = 1
        return 0
            

    def check(self, x, y, last_x, last_y, dir):
        self.count += 1
        if self.count >= 5:
            return True
        for i in range(2):
            new_x = x + self.check_dir[dir][i][0]
            new_y = y + self.check_dir[dir][i][1]
            print(new_x, " ", new_y, self.dimention, last_x, last_y)
            if ((0 <= new_x < self.dimention) and (0 <= new_y < self.dimention)) and not (new_x == last_x and new_y == last_y):
                print("  ", new_x, " ", new_y)
                if self.board[new_x][new_y] == self.color:
                    if self.check(new_x, new_y, x, y, dir):
                        return True

test = game(lim= 0)
state = 0
while not (state == -2):
    x = int(input())
    y = int(input())
    state = test.play(x, y)
    print(state)
    print(test.board)
exit(0)