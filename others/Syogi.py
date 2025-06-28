from tkinter import messagebox
import numpy as np
import os
import sys
import pygame

class Syogi:
    def __init__(self):
        self.Y, self.X = 1000, 1000
        self.X_GAP = 300
        pygame.init()
        self.screen = pygame.display.set_mode((self.X + self.X_GAP * 2, self.Y))
        pygame.display.set_caption("Syogi Battle")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.piece = pygame.image.load(os.path.join(self.base_dir, 'images', 'syogi_pieces.png')).convert_alpha()
        self.PIECE_SIZE = min(self.X / self.piece.get_width() * 7 / 9, self.Y / self.piece.get_height() * 7 / 9)
        self.piece = pygame.transform.scale(self.piece, (self.piece.get_width() * self.PIECE_SIZE, self.piece.get_height() * self.PIECE_SIZE))
        self.PIECES_IMG_SIZE_X, self.PIECES_IMG_SIZE_Y = self.piece.get_width(), self.piece.get_height()

        self.piece_img_pos = [(self.PIECES_IMG_SIZE_X - (self.PIECES_IMG_SIZE_X / 8) * i, 0, (self.PIECES_IMG_SIZE_X / 8), (self.PIECES_IMG_SIZE_Y / 4)) for i in range(9)]
        self.piece_img_pos.append((0, (self.PIECES_IMG_SIZE_Y / 4), (self.PIECES_IMG_SIZE_X / 8), (self.PIECES_IMG_SIZE_Y / 4)))

        self.initialize()
        

    def initialize(self):

        self.state = np.array([
            [[2, 1, 0], [3, 1, 0], [4, 1, 0], [5, 1, 0], [9, 1, 0], [5, 1, 0], [4, 1, 0], [3, 1, 0], [2, 1, 0]],
            [[0, -1, 0], [7, 1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [6, 1, 0], [0, -1, 0]],
            [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]],
            [[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]],
            [[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]],
            [[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]],
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, -1, 0], [6, 0, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [7, 0, 0], [0, -1, 0]],
            [[2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [8, 0, 0], [5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0]]
        ])

        self.already_chosen = False
        self.chosen_motigoma = -1
        self.ables = []
        self.last_clicked = []

        self.piece_in_hands = [[], []] #under, upper

        self.turn = 1 # 1:under, -1: upper

        self.mouse_clicked = False

        self.is_gameend = False
        
    def get_selectable(self, piece_type, is_naru): # (歩、香車、桂馬、銀、金、角、飛車、玉、王)
        pieces = [(1, 0), # 各駒の動ける方向
                [[(i, 0) for i in range(1, 9)]],
                [[(2, 1)], [(2, -1)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(-1, -1)], [(-1, 1)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(0, 1)], [(0, -1)], [(-1, 0)]],
                [[(i, i)  for i in range(-1, -9, -1)], [(i, i) for i in range(1, 9)], [(-i, i)  for i in range(-1, -9, -1)], [(-i, i) for i in range(1, 9)]],
                [[(i, 0)  for i in range(-1, -9, -1)], [(i, 0) for i in range(1, 9)], [(0, i)  for i in range(-1, -9, -1)], [(0, i) for i in range(1, 9)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(0, -1)], [(0, 1)], [(-1, 0)], [(-1, -1)], [(-1, 1)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(0, -1)], [(0, 1)], [(-1, 0)], [(-1, -1)], [(-1, 1)]],
                ]
        naru_pieces = [[[(1, 0)], [(1, 1)], [(1, -1)], [(0, 1)], [(0, -1)], [(-1, 0)]], # 各駒の動ける方向
                [[[(1, 0)], [(1, 1)], [(1, -1)], [(0, 1)], [(0, -1)], [(-1, 0)]]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(0, 1)], [(0, -1)], [(-1, 0)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(0, 1)], [(0, -1)], [(-1, 0)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(0, 1)], [(0, -1)], [(-1, 0)]],
                [[(i, i)  for i in range(-1, -9, -1)], [(i, i) for i in range(1, 9)], [(-i, i)  for i in range(-1, -9, -1)], [(-i, i) for i in range(1, 9)], [(1, 0)], [(-1, 0)], [(0, 1)], [(0, -1)]],
                [[(i, 0)  for i in range(-1, -9, -1)], [(i, 0) for i in range(1, 9)], [(0, i)  for i in range(-1, -9, -1)], [(0, i) for i in range(1, 9)], [(1, 1)], [(1, -1)], [(-1, -1)], [(-1, 1)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(0, -1)], [(0, 1)], [(-1, 0)], [(-1, -1)], [(-1, 1)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(0, -1)], [(0, 1)], [(-1, 0)], [(-1, -1)], [(-1, 1)]],
                ]
        return pieces[piece_type] if not is_naru else naru_pieces[piece_type]

    def check_selectable(self, pos):
        able_pos = []
        piece = int(self.state[pos][0] - 1)
        if piece < 0:
            return []
        pos_list = self.get_selectable(piece_type=piece, is_naru=self.state[pos][2])
        if type(pos_list[0]) != list:
            
            coordinate = pos + np.array(pos_list) * (-1) ** (self.state[pos][1] + 1)
            if 0 <= coordinate[0] <= 8 and 0 <= coordinate[1] <= 8 and self.state[pos[0], pos[1], 1] != self.state[coordinate[0], coordinate[1], 1]:
                able_pos.append(coordinate)
            return able_pos
        
        for way in pos_list:
            for move_index in way:
                coordinate = pos + np.array(move_index) * (-1) ** (self.state[pos][1] + 1)
                if 0 <= coordinate[0] <= 8 and 0 <= coordinate[1] <= 8 and self.state[coordinate[0], coordinate[1], 0] == 0 and self.state[pos[0], pos[1], 1] != self.state[coordinate[0], coordinate[1], 1]:
                    able_pos.append(coordinate)
                else:
                    if 0 <= coordinate[0] <= 8 and 0 <= coordinate[1] <= 8 and self.state[pos[0], pos[1], 1] != self.state[coordinate[0], coordinate[1], 1]:
                        able_pos.append(coordinate)
                    break
        return able_pos
    
    def draw_state_and_put_motigoma(self, screen):
        for i in range(10):
            pygame.draw.line(screen, (0, 0, 0), (self.X_GAP + self.X * i / 9, 0), (self.X_GAP + self.X * i / 9, self.Y), 8)
            pygame.draw.line(screen, (0, 0, 0), (self.X_GAP, self.Y * i / 9), (self.X_GAP + self.X, self.Y * i / 9), 8)
        for j in range(9):
            for k in range(9):
                if self.state[k, j, 0] >= 0:
                    img_pos = self.piece_img_pos[int(self.state[k, j, 0])]
                    if self.state[k, j, 0] == 4 or self.state[k, j, 0] == 4 or self.state[k, j, 2] == 0:
                        clip_rect = pygame.Rect(img_pos)
                    else:
                        clip_rect = pygame.Rect((img_pos[0], img_pos[1] + self.PIECES_IMG_SIZE_Y / 4, img_pos[2], img_pos[3]))
                    # 指定した部分だけを新しいSurfaceにコピー
                    cropped = pygame.Surface((clip_rect.width, clip_rect.height), pygame.SRCALPHA)
                    cropped.blit(self.piece, (0, 0), area=clip_rect)

                    # 上下反転（flip_y=True, flip_x=False）

                    flipped = pygame.transform.flip(cropped, False, int(self.state[k, j, 1]))
                    
                    screen.blit(flipped, (self.X_GAP + self.X * (j / 9 + 1 / 126), self.Y * (k / 9 + 1 / 252)))
        for pos in self.ables:
            pygame.draw.circle(screen, (255,0,0), (self.X_GAP + pos[1] * self.X / 9 + self.X / 18, pos[0] * self.Y / 9 + self.Y / 18), 5, 5)
        
        for player, player_has in enumerate(self.piece_in_hands):
            for index, piece in enumerate(player_has):
                
                clip_rect = pygame.Rect(self.piece_img_pos[piece])
                
                # 指定した部分だけを新しいSurfaceにコピー
                cropped = pygame.Surface((clip_rect.width, clip_rect.height), pygame.SRCALPHA)
                cropped.blit(self.piece, (0, 0), area=clip_rect)
                # 上下反転（flip_y=True, flip_x=False）

                flipped = pygame.transform.flip(cropped, False, player)
                
                screen.blit(flipped, ((1 - player) * (self.X_GAP + self.X) + self.X_GAP / 8 * (4 * ((index + player) % 2) + 1), player * self.Y * 8 / 9 + self.Y / 9 * (8 - index // 2) * (-1) ** player))
                mouse_pos = pygame.mouse.get_pos()
                if (abs((1 - player) * (self.X_GAP + self.X) + self.X_GAP / 8 * (4 * ((index + player) % 2) + 1) + self.PIECES_IMG_SIZE_X / 16 - mouse_pos[0]) <= self.PIECES_IMG_SIZE_X / 8 and 
                    abs(player * self.Y * 8 / 9 + self.Y / 9 * (8 - index // 2) * (-1) ** player + self.PIECES_IMG_SIZE_Y / 8 - mouse_pos[1]) <= self.PIECES_IMG_SIZE_Y / 4 and 
                    self.mouse_clicked and (1 - player * 2) == self.turn):
                    self.ables = []
                    if piece == 1:
                        for row in range(9):
                            if np.any(np.all(np.array([1, player]) == self.state[:, row, 0:2], axis=1)):
                                continue
                            for i, j in enumerate(self.state[:, row, 0]):
                                if j == 0:
                                    self.ables.append([i, row])
                                    self.chosen_motigoma = piece
                    else:
                        for row in range(9):
                            for i, j in enumerate(self.state[:, row, 0]):
                                if j == 0:
                                    self.ables.append([i, row])
                                    self.chosen_motigoma = piece
                
                    
                    

    
    def put_piece(self, old_pos, pos):
        if int(self.state[pos[0], pos[1], 1]) != -1:
            self.piece_in_hands[int((self.turn - 1)/2)].append(int(self.state[pos[0], pos[1], 0]))
            if int(self.state[pos[0], pos[1], 0]) >= 8:
                self.is_gameend = True
        
        self.state[pos[0], pos[1]], self.state[old_pos[0], old_pos[1]] = self.state[old_pos[0], old_pos[1]], np.array([0, -1, 0])
        if ((self.turn == 1 and pos[0] <= 2) or (self.turn == -1 and pos[0] >= 6)) and self.state[pos[0], pos[1], 2] == 0 and not (self.state[pos[0], pos[1], 0] == 5 or self.state[pos[0], pos[1], 0] >= 8):
            if messagebox.askyesno("確認", "成りますか?"):
                self.state[pos[0], pos[1], 2] = 1
        if np.any(np.all(self.ables == np.array(pos), axis=1)) and self.already_chosen:
            self.turn *= -1
        
        
        self.last_clicked = []
        self.already_chosen = False
        self.ables = []
    
    def put_motigoma(self, pos):
        self.state[pos[0], pos[1]] = [self.chosen_motigoma, (1 - self.turn) / 2, 0]
        self.ables = []
        self.piece_in_hands[int((1 - self.turn) / 2)] = []
        self.turn *= -1
        self.chosen_motigoma = -1

    def clicked(self):
        mouse_pos = pygame.mouse.get_pos()
        mouse_x, mouse_y = mouse_pos[0] - self.X_GAP, mouse_pos[1]

        if not (0 <= int(mouse_y // (self.Y / 9)) <= 8 and 0 <= int(mouse_x // (self.X / 9)) <= 8):
            return
        
        if self.chosen_motigoma != -1 and len(self.ables) > 0 and np.any(np.all(self.ables == np.array([int(mouse_y // (self.Y / 9)), int(mouse_x // (self.X / 9))]), axis=1)):
            self.put_motigoma([int(mouse_y // (self.Y / 9)), int(mouse_x // (self.X / 9))])
            return

        if (((not self.already_chosen) and self.state[int(mouse_y // (self.Y / 9)), int(mouse_x // (self.X / 9)), 1] != (self.turn + 1) / 2) or 
            (self.already_chosen and (np.any(np.all(self.ables == np.array([int(mouse_y // (self.Y / 9)), int(mouse_x // (self.X / 9))]), axis=1)) or 
            self.state[int(mouse_y // (self.Y / 9)), int(mouse_x // (self.X / 9)), 1] == (1 - self.turn)/2))):
            if self.already_chosen and np.any(np.all(self.ables == np.array([int(mouse_y // (self.Y / 9)), int(mouse_x // (self.X / 9))]), axis=1)):
                self.put_piece(self.last_clicked, (int(mouse_y // (self.Y / 9)), int(mouse_x // (self.X / 9))))
                return
            
            
            able_pos = self.check_selectable((int(mouse_y // (self.Y / 9)), int(mouse_x // (self.X / 9))))
            self.ables = able_pos

            self.last_clicked = [int(mouse_y // (self.Y / 9)), int(mouse_x // (self.X / 9))]

            if len(self.ables) > 0:
                self.already_chosen = True
            else:
                self.already_chosen = False

    def step(self, old_pos=[-1, -1], pos=[-1, -1], use_motigoma=-1, motigoma_pos=[-1, -1]): # 0:歩, 1:香車 2:桂馬, 3:銀, 4:金, 5:角, 6:飛車, 7:玉, 8:王
        if pos == [-1, -1]:
            self.chosen_motigoma = use_motigoma
            if self.state[motigoma_pos][0] == 0:
                self.put_motigoma(motigoma_pos)
                reward = len(self.ables[int((1 - self.turn) / 2)])
            else:
                reward = -1
        else:
            self.put_piece(old_pos, pos)
        
        
        return self.state, self.piece_in_hands, self.turn, self.is_gameend, reward


    def main(self):
        # フレームレート制御用のクロック
        clock = pygame.time.Clock()
        FPS = 60  # フレーム毎秒

        # 色定義（RGB）
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)

        # メインループ
        running = True
        while running:
            # イベント処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_clicked = True
                    self.clicked()
                else:
                    self.mouse_clicked = False

            # 画面の塗りつぶし
            self.screen.fill(BLACK)

            # --- ここに描画処理を書く ---

            back = pygame.image.load(os.path.join(self.base_dir, 'images', 'syogi_back.png')).convert_alpha()
            back = pygame.transform.scale(back, (self.X + self.X_GAP * 2, self.Y))
            self.screen.blit(back ,(0, 0))

            

            self.draw_state_and_put_motigoma(self.screen)
            # 画面更新
            pygame.display.flip()

            # FPS制限
            clock.tick(FPS)

        # 終了処理
        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    syogi = Syogi()
    syogi.main()

