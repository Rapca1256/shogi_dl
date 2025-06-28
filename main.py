import numpy as np
import multiprocessing
import itertools
import pygame
import sys
import os

class Syogi:
    def __init__(self):
        self.reset()
        #print(self.get_learning_data())
    
    def reset(self):
        # [駒の種類, 誰が持っているか, 成っているか]
        self.state = np.array([
            [[2, -1, 0], [3, -1, 0], [4, -1, 0], [5, -1, 0], [9, -1, 0], [5, -1, 0], [4, -1, 0], [3, -1, 0], [2, -1, 0]],
            [[0, 0, 0], [7, -1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [6, -1, 0], [0, 0, 0]],
            [[1, -1, 0], [1, -1, 0], [1, -1, 0], [1, -1, 0], [1, -1, 0], [1, -1, 0], [1, -1, 0], [1, -1, 0], [1, -1, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]],
            [[0, 0, 0], [6, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [7, 1, 0], [0, 0, 0]],
            [[2, 1, 0], [3, 1, 0], [4, 1, 0], [5, 1, 0], [8, 1, 0], [5, 1, 0], [4, 1, 0], [3, 1, 0], [2, 1, 0]]
        ])
        self.motigoma = [[np.zeros(3) for _ in range(27)] for _ in range(2)]

        self.turn = 1

        self.selecting = [0, 0] # 0は通常時、1は歩を選択中(二歩が不可)、2はそれ以外を選択中
        self.selecting_motigoma = [0, 0] # 0:F, 1:T
        self.selected_pos = [0, 0, 0] #index0:下/上、index1,2:持ち駒中の座標(y, x)



    def get_selectable(self, piece_type, is_naru): # (歩、香車、桂馬、銀、金、角、飛車、玉、王)
        # 通常時の駒の動き
        pieces = [[[(1, 0)]],
                [[(i, 0) for i in range(1, 9)]],
                [[(2, 1)], [(2, -1)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(-1, -1)], [(-1, 1)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(0, 1)], [(0, -1)], [(-1, 0)]],
                [[(i, i)  for i in range(-1, -9, -1)], [(i, i) for i in range(1, 9)], [(-i, i)  for i in range(-1, -9, -1)], [(-i, i) for i in range(1, 9)]],
                [[(i, 0)  for i in range(-1, -9, -1)], [(i, 0) for i in range(1, 9)], [(0, i)  for i in range(-1, -9, -1)], [(0, i) for i in range(1, 9)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(0, -1)], [(0, 1)], [(-1, 0)], [(-1, -1)], [(-1, 1)]],
                [[(1, 0)], [(1, 1)], [(1, -1)], [(0, -1)], [(0, 1)], [(-1, 0)], [(-1, -1)], [(-1, 1)]],
                ]
        
        # 成ったときのやつ
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
        return pieces[int(piece_type - 1)] if not is_naru else naru_pieces[int(piece_type - 1)]
    

    def get_ables(self, pos:np.array):
        turn = self.turn
        if self.selecting_motigoma[(1 - turn) // 2] == 0:
            ables = []
            piece_type, piece_whose, piece_naru = self.state[pos[0], pos[1]]
            directions = self.get_selectable(piece_type, piece_naru)
            for moves in directions:
                for move in moves:
                    move_to = np.array(move) * ((-1) ** ((turn + 1) / 2)) + np.array(pos)
                    try:
                        if 0 <= move_to[0] < 9 and 0 <= move_to[1] < 9:
                            ables.append(move_to)
                    except Exception as e:
                        print(e, move_to[0], move_to[1])
                        raise Exception
            return ables
        elif self.selecting_motigoma[(1 - turn) // 2] == 1 and self.state[pos[0], pos[1]][0] == 1:
            ables = []
            for i in range(9):
                if 1 in self.state[i, :] or not 0 in self.state[i, :]:
                    continue
                ables += [[i, j] for i in range(9) for j in range(9) if self.state[i, j, 0] == 0]
            return ables
        else:
            ables = []
            for i in range(9):
                if not 0 in self.state[i, :]:
                    continue
                ables += [[i, j] for i in range(9) for j in range(9) if self.state[i, j, 0] == 0]
            return ables
    
    def get_legal_action(self, turn):
        legal_action = []

        for y in range(9):
            for x in range(9):
                piece = self.state[y, x]
                if piece[1] == turn:
                    ables = self.get_ables(np.array([y, x]))
                    for to_y, to_x in ables:
                        action = {
                            "motigoma_use" : 0,
                            "koma_from" : np.array([y, x]),
                            "koma_to" : np.array([to_y, to_x]),
                            "motigoma_type" : 0,
                            "motigoma_pos" : np.array([0, 0]),
                            "wanna_naru" : 0
                        }
                        legal_action.append(action)
                        if abs(to_y - 4) >= 2:
                            action = {
                                "motigoma_use" : 0,
                                "koma_from" : np.array([y, x]),
                                "koma_to" : np.array([to_y, to_x]),
                                "motigoma_type" : 0,
                                "motigoma_pos" : np.array([0, 0]),
                                "wanna_naru" : 1
                            }
                            legal_action.append(action)

        for piece_type in range(1, 10):
            if self.deform_motigoma()[(1 - turn) // 2][piece_type] > 0:
                for y in range(9):
                    for x in range(9):
                        if piece_type == 1 and np.any(np.all(np.array([piece_type, turn, 0]) == self.state[:, x], axis=1)):
                            continue

                        if self.state[y, x][0] == 0:
                            action = {
                                "motigoma_use": 1,
                                "koma_from" : np.array([y, x]),
                                "koma_to" : np.array([to_y, to_x]),
                                "motigoma_type": piece_type,
                                "motigoma_pos": [y, x],
                                "wanna_naru" : 0
                            }
                            legal_action.append(action)
        
        return legal_action






    def put_koma(self, koma_from, koma_to, naru=0):
        turn = self.turn

        # 有効な範囲内か確認
        if not (0 <= koma_from[0] <= 8 or 0 <= koma_from[1] <= 8 or 0 <= koma_to[0] <= 8 or 0 <= koma_to[1] <= 8):
            return False
        
        # 移動する駒の情報取得
        piece_type, piece_whose, piece_naru = self.state[koma_from[0], koma_from[1]]
        
        # 動かす駒が自分のじゃなければ帰る
        if piece_whose != turn:
            return False
        
        # 取れる位置をすべて取得
        ables = self.get_ables(koma_from)

        for check_pos in ables:
            # 可能な位置一覧にあって自分の駒でないなら移動できる
            if np.any(np.all(check_pos == np.array(koma_to), axis=0)) and self.state[koma_to[0]][koma_to[1]][1] / turn != 1:
                # 相手の駒なら取って自分の持ち駒にする
                if self.state[koma_to[0]][koma_to[1]][1] / turn == -1:
                    add_motigoma_indexs = list(map(int, itertools.chain.from_iterable(self.motigoma[int((1 - turn) / 2)]))).index(0)
                    self.motigoma[int((1 - turn) / 2)][int(add_motigoma_indexs)] = self.state[koma_to[0]][koma_to[1]]
                    #print(int((1 - turn) / 2), int(add_motigoma_indexs // 3), int(add_motigoma_indexs % 3))
                    #print('tottayo!', self.state[koma_to[0]][koma_to[1]])
                self.state[koma_to[0]][koma_to[1]], self.state[koma_from[0]][koma_from[1]] = self.state[koma_from[0]][koma_from[1]], np.array([0, 0, 0])
                
                if naru == 1 and self.state[koma_to[0]][koma_to[1]][2] == 0 and ((koma_to[0] <= 2 and self.state[koma_to[0]][koma_to[1]][1] == 1) or (koma_to[0] >= 6 and self.state[koma_to[0]][koma_to[1]][1] == -1)):
                    self.state[koma_to[0]][koma_to[1]][2] = 1
            
                return True
        return False

    def put_motigoma(self, piece, pos):
        turn = self.turn

        # 2歩処理
        if piece == 1 and np.any(np.all(np.array([piece, turn, 0]) == self.state[:, pos[1]], axis=1)):
            return False
        
        #その場所に駒がなければ置ける
        if self.state[pos[0], pos[1]][0] == 0:
            index = np.argwhere(self.motigoma[int((1 - turn) / 2)] == piece)
            if index.size==0:
                return False
            self.state[pos[0], pos[1]] = np.array([piece, turn, 0])
            try:
                self.motigoma[int((1 - turn) / 2)][index[0]][index[1]] = 0
            except Exception as e:
                print(e)
                print(index)

                raise Exception

            return True
        return False

    def step(self, motigoma_use=False, koma_from=None, koma_to=None, motigoma_type=None, motigoma_pos=None, wanna_naru = 0):
        if motigoma_use:
            result = self.put_motigoma(motigoma_type, motigoma_pos)
            done = self.check_win()
            if result:
                self.turn *= -1
            return self.state, self.motigoma, self.turn, result, done
        result = self.put_koma(koma_from=koma_from, koma_to=koma_to, naru=wanna_naru)
        if result:
            self.turn *= -1
        done, is_win = self.check_win()
        if done:
            winner = is_win.index(1)
        else:
            winner = None
        return self.state, self.motigoma, self.turn, result, done, winner
    
    def check_win(self):
        check_list = {8: 0, 9: 0}
        for line in self.state[:, :, 0]:
            if 8 in line:
                check_list[8] = 1
            if 9 in line:
                check_list[9] = 1
        if check_list[8] and check_list[9]:
            return False, check_list.values()
        return True, check_list.values()
    
    def get_info(self):
        return self.state, self.motigoma, self.turn
    
    def print_state(self):
        
        
        print('#######################')
        for row in self.state:
            print(''.join(map(str, map(get_list_element, row, [0 for _ in range(len(row))]))))

        print('')
        
        datas = self.get_learning_data()
        for data in datas:
            print('')
            for row in data:

                print(''.join(map(str, map(int, row))))

        print('')

        print(datas)

    
    def deform_motigoma(self):
        deformed_motigoma = []
        for row in reversed(self.motigoma):
            d = motigoma_to_dict(row)
            deformed_motigoma.append(d)
        return deformed_motigoma
    
    def get_learning_data(self):
        data_1 = np.zeros((10, 9))
        data_2 = np.zeros((10, 9))
        
        for i, row in enumerate(self.state):
            for n, piece in enumerate(row):
                if piece[1] == 1:
                    data_1[i, n] = piece[0] * (piece[2]*9 + 1)
                elif piece[1] == -1:
                    data_2[i, n] = piece[0] * (piece[2]*9 + 1)
        
        for k, num in enumerate(self.deform_motigoma()[0].values()):
            data_1[9, k] = num

        for k, num in enumerate(self.deform_motigoma()[1].values()):
            data_2[9, k] = num

        if self.turn == 1:
            return np.array([data_1, data_2])
        else:
            return np.array([data_2, data_1])
    
    def act_to_scalar(self, act):
        # POSITION : 0 ~ 80
        # MOTIGOMA_TYPE : 1 ~ 7
        # NO_MOTIGOMA = (koma_from) * (koma_to) * (wanna_naru) = 81 * 81 * 2 = 13122
        # WITH_MOTIGOMA = (motigoma_type) * (motigoma_to) = 7 * 81 = 567 <- 王、玉は持ち駒にならないことに注意
        # SUM = 13122 + 567 = 13689
        if type(act) == dict:
            act = act.values()
        motigoma_use, koma_from, koma_to, motigoma_type, motigoma_pos, wanna_naru = act
        assert motigoma_type < 8, ValueError('motigoma_type must be less than 8.')
        koma_from_pos = koma_from[0] * 9 + koma_from[1]
        koma_to_pos = koma_to[0] * 9 + koma_to[1]
        motigoma_pos_flatten = motigoma_pos[0] * 9 + motigoma_pos[1]
        
        if motigoma_use == 0:  # 普通の移動
            score = (koma_from_pos * 81 + koma_to_pos) * 2 + wanna_naru
        else:  # 持ち駒打ち
            score = 13122 + (motigoma_type - 1) * 81 + motigoma_pos_flatten

        return int(score)
    
    def scalar_to_act(self, scalar):
        if scalar <= 13121:
            # 通常の移動手
            wanna_naru = scalar % 2
            move_index = scalar // 2
            koma_from_index = move_index // 81
            koma_to_index = move_index % 81

            koma_from = (koma_from_index // 9, koma_from_index % 9)
            koma_to = (koma_to_index // 9, koma_to_index % 9)

            motigoma_use = 0
            motigoma_type = 0
            motigoma_pos = (0, 0)

        else:
            # 持ち駒打ち
            idx = scalar - 13122
            motigoma_type = idx // 81 + 1  # +1 で 1〜7 に戻す
            motigoma_pos_flatten = idx % 81
            motigoma_pos = (motigoma_pos_flatten // 9, motigoma_pos_flatten % 9)

            koma_from = 0
            koma_to = (motigoma_pos_flatten // 9, motigoma_pos_flatten % 9)
            wanna_naru = 0
            motigoma_use = 1

        return (motigoma_use, koma_from, koma_to, motigoma_type, motigoma_pos, wanna_naru)



def get_list_element(list, index):
    return list[index]

def motigoma_to_dict(motigoma):
    dictionary = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    for i in motigoma:
        dictionary[int(i[0])] += 1
    
    dictionary.pop(0)
    
    return dictionary




class DrawState:
    def __init__(self, syogi):
        self.Y, self.X = 900, 900
        self.X_GAP = 300
        pygame.init()
        self.screen = pygame.display.set_mode((self.X + self.X_GAP * 2, self.Y)) # (5:3)
        pygame.display.set_caption("Syogi Battle")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.piece = pygame.image.load(os.path.join(self.base_dir, 'images', 'syogi_pieces.png')).convert_alpha()
        self.PIECE_SIZE = min(self.X / self.piece.get_width() * 7 / 9, self.Y / self.piece.get_height() * 7 / 9)
        self.piece = pygame.transform.scale(self.piece, (self.piece.get_width() * self.PIECE_SIZE, self.piece.get_height() * self.PIECE_SIZE))
        self.PIECES_IMG_SIZE_X, self.PIECES_IMG_SIZE_Y = self.piece.get_width(), self.piece.get_height()

        self.piece_img_pos = [(self.PIECES_IMG_SIZE_X - (self.PIECES_IMG_SIZE_X / 8) * i, 0, (self.PIECES_IMG_SIZE_X / 8), (self.PIECES_IMG_SIZE_Y / 4)) for i in range(9)]
        self.piece_img_pos.append((0, (self.PIECES_IMG_SIZE_Y / 4), (self.PIECES_IMG_SIZE_X / 8), (self.PIECES_IMG_SIZE_Y / 4)))

        self.syogi = syogi
    
    def draw_state(self, state, motigoma):
        screen = self.screen
        for i in range(10):
            pygame.draw.line(screen, (0, 0, 0), (self.X_GAP + self.X * i / 9, 0), (self.X_GAP + self.X * i / 9, self.Y), 8)
            pygame.draw.line(screen, (0, 0, 0), (self.X_GAP, self.Y * i / 9), (self.X_GAP + self.X, self.Y * i / 9), 8)
        for j in range(9):
            for k in range(9):
                if state[k, j, 0] >= 0:
                    img_pos = self.piece_img_pos[int(state[k, j, 0])]
                    if state[k, j, 0] == 4 or state[k, j, 0] == 4 or state[k, j, 2] == 0:
                        clip_rect = pygame.Rect(img_pos)
                    else:
                        clip_rect = pygame.Rect((img_pos[0], img_pos[1] + self.PIECES_IMG_SIZE_Y / 4, img_pos[2], img_pos[3]))
                    # 指定した部分だけを新しいSurfaceにコピー
                    cropped = pygame.Surface((clip_rect.width, clip_rect.height), pygame.SRCALPHA)
                    cropped.blit(self.piece, (0, 0), area=clip_rect)

                    # 上下反転（flip_y=True, flip_x=False）

                    flipped = pygame.transform.flip(cropped, False, int(state[k, j, 1] - 1))
                    
                    screen.blit(flipped, (self.X_GAP + self.X * (j / 9 + 1 / 126), self.Y * (k / 9 + 1 / 252)))
        #for pos in self.ables:
        #    pygame.draw.circle(screen, (255,0,0), (self.X_GAP + pos[1] * self.X / 9 + self.X / 18, pos[0] * self.Y / 9 + self.Y / 18), 5, 5)
        
        for player, player_has in enumerate(motigoma):
            for index, piece in enumerate([i for i in list(itertools.chain.from_iterable(player_has)) if i != 0]):
                piece = int(piece)
                clip_rect = pygame.Rect(self.piece_img_pos[piece])
                
                # 指定した部分だけを新しいSurfaceにコピー
                cropped = pygame.Surface((clip_rect.width, clip_rect.height), pygame.SRCALPHA)
                cropped.blit(self.piece, (0, 0), area=clip_rect)
                # 上下反転（flip_y=True, flip_x=False）

                flipped = pygame.transform.flip(cropped, False, player)
                
                screen.blit(flipped, ((1 - player) * (self.X_GAP + self.X) + self.X_GAP / 3 * ((index + player) % 3), player * self.Y * 8 / 9 + self.Y / 9 * (8 - index // 3) * (-1) ** player))
    
    def clicked(self, pos: np.array):
        y, x = pos // (self.Y // 9)
        if 0 <= x <= 2:
            syogi.selecting_motigoma[(1 - syogi.turn) // 2] = 1
            # 上のプレイヤーの持ち駒

            if syogi.motigoma[1][y][x] == 1:
                syogi.selecting[(1 - syogi.turn) // 2] = 1
                syogi.selected_pos = [1, y, x]
            
            if syogi.motigoma[1][y][x] > 1:
                syogi.selecting[(1 - syogi.turn) // 2] = 2
                syogi.selected_pos = [2, y, x]

        elif 3 <= x <= 11:
            syogi.selecting_motigoma[(1 - syogi.turn) // 2] = 0
            # 盤面
            if syogi.selecting[(1 - syogi.turn) // 2] != 0:
                if 3 <= syogi.selected_pos[2] <= 11:
                    syogi.step(False, koma_from=[syogi.selected_pos[1], syogi.selected_pos[2] - 3], koma_to=np.array([y, x - 3]))
                elif 0 <= syogi.selected_pos[2] <= 2:
                    if syogi.turn == 1:
                        return
                    syogi.step(motigoma_use=True, motigoma_type=syogi.motigoma[syogi.selected_pos[1]][syogi.selected_pos[2]][0], motigoma_pos=np.array([y, x - 3]))
                else:
                    if syogi.turn == -1:
                        return
                    syogi.step(motigoma_use=True, motigoma_type=syogi.motigoma[syogi.selected_pos[1]][syogi.selected_pos[2] - 12][0], motigoma_pos=np.array([y, x - 3]))
                syogi.selecting[(1 - syogi.turn) // 2] = 0
            else:
                syogi.selecting_motigoma[(1 - syogi.turn) // 2] = 1
                if syogi.state[y, x - 3, 0] == 1:
                    syogi.selecting[(1 - syogi.turn) // 2] = 1
                else:
                    syogi.selecting[(1 - syogi.turn) // 2] = 2
                syogi.selected_pos = [(1 - syogi.turn) // 2, y, x]

            

        
        else:
            # 下のプレイヤーの持ち駒
            if syogi.motigoma[0][8 - y][x - 12] == 1:
                syogi.selecting[(1 - syogi.turn) // 2] = 1
                syogi.selected_pos = [1, 8 - y, x]
            
            if syogi.motigoma[0][8 - y][x - 12] > 1:
                syogi.selecting[(1 - syogi.turn) // 2] = 2
                syogi.selected_pos = [2, 8 - y, x]

    def pygame_main(self):
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
                    mousex, mousey = pygame.mouse.get_pos()
                    self.clicked(np.array([mousey, mousex]))
                else:
                    self.mouse_clicked = False

            # 画面の塗りつぶし
            self.screen.fill(BLACK)

            # --- ここに描画処理を書く ---

            back = pygame.image.load(os.path.join(self.base_dir, 'images', 'syogi_back.png')).convert_alpha()
            back = pygame.transform.scale(back, (self.X + self.X_GAP * 2, self.Y))
            self.screen.blit(back ,(0, 0))

            
            self.draw_state(syogi.state, syogi.motigoma)

            # 画面更新
            pygame.display.flip()

            # FPS制限
            clock.tick(FPS)

        # 終了処理
        pygame.quit()
        sys.exit()


def str_to_bool(str):
    if not str in ['True', 'False']:
        return None
    return True if str == 'True' else False

def game(syogi):
    state, motigoma, turn = syogi.get_info()
    syogi.print_state()
    while True:
        print(syogi.motigoma)
        print(motigoma_to_dict(syogi.motigoma[0]))
        print(motigoma_to_dict(syogi.motigoma[1]))
        actions = input('turn:{} move:'.format(turn)).split(' ')
        try:
            y, x ,to_y, to_x, motigoma, naru = map(int, actions[1:]) # 持ち駒を使うならy, xはNone
            use_motigoma = str_to_bool(actions[0])
            state, motigoma, turn, result, done = syogi.step(motigoma_use=use_motigoma, koma_from=(y, x), koma_to=(to_y, to_x), motigoma_type=motigoma, motigoma_pos=(to_y, to_x), wanna_naru=naru)
            #draw.draw_state(state, motigoma)
            syogi.print_state()
        except ValueError:
            syogi.print_state()

if __name__ == '__main__':
    syogi = Syogi()
    #draw = DrawState(syogi)
    
    for i in range(2):
        syogi.step(False, koma_from=(6, i), koma_to=(5, i))
        syogi.step(False, koma_from=(2, i), koma_to=(3, i))
        syogi.step(False, koma_from=(5, i), koma_to=(4, i))
        syogi.step(False, koma_from=(3, i), koma_to=(4, i))
    #print(syogi.step(False, koma_from=(6, 6), koma_to=(5, 6)))
    
    #print(syogi.step(False, koma_from=(2, 4), koma_to=(3, 4)))
    #print(syogi.step(False, koma_from=(6, 4), koma_to=(5, 4)))
    #syogi.step(False, koma_from=(3, 4), koma_to=(4, 4))
    #syogi.step(False, koma_from=(5, 4), koma_to=(4, 4))
    #for i in range(9):
        #for j in range(3):
            #syogi.motigoma[0][i][j] = 8
    print(syogi.turn)
    print(syogi.get_legal_action(syogi.turn))

    game(syogi=syogi)
