import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import os
import copy

class Syogi:
    def __init__(self):
        self.reset()

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
        
        # 持ち駒を辞書形式で管理（より簡潔）
        self.motigoma = [{i: 0 for i in range(1, 10)} for _ in range(2)]
        
        self.turn = 1
        self.position_history = []
        self.max_turn_count = 300
        self.turn_count = 0

    def get_player_index(self, turn):
        """プレイヤーのインデックスを取得（1 -> 0, -1 -> 1）"""
        return 0 if turn == 1 else 1

    def can_promote(self, piece_type):
        """駒が成れるかどうかを判定"""
        # 金、王、玉、成り駒は成れない
        return piece_type in [1, 2, 3, 4, 6, 7]

    def get_selectable(self, piece_type, is_naru):
        """駒の移動可能な方向を取得（方向ベース）"""
        # 通常時の駒の動き（方向ベース）
        pieces = [
            [(1, 0)],  # 歩：前に1マス
            [(1, 0)],  # 香車：前方向に直進
            [(2, 1), (2, -1)],  # 桂馬：特殊な動き
            [(1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)],  # 銀：5方向に1マス
            [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, 0)],  # 金：6方向に1マス
            [(1, 1), (1, -1), (-1, 1), (-1, -1)],  # 角：斜め4方向に直進
            [(1, 0), (-1, 0), (0, 1), (0, -1)],  # 飛車：縦横4方向に直進
            [(1, 0), (1, 1), (1, -1), (0, -1), (0, 1), (-1, 0), (-1, -1), (-1, 1)],  # 玉：8方向に1マス
            [(1, 0), (1, 1), (1, -1), (0, -1), (0, 1), (-1, 0), (-1, -1), (-1, 1)],  # 王：8方向に1マス
        ]

        # 成ったときの動き
        naru_pieces = [
            [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, 0)],  # と金：金と同じ
            [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, 0)],  # 成香：金と同じ
            [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, 0)],  # 成桂：金と同じ
            [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, 0)],  # 成銀：金と同じ
            [(1, 0), (1, 1), (1, -1), (0, 1), (0, -1), (-1, 0)],  # 金：変化なし
            [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)],  # 竜馬：角+王の動き
            [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, -1), (-1, 1)],  # 竜王：飛車+王の動き
            [(1, 0), (1, 1), (1, -1), (0, -1), (0, 1), (-1, 0), (-1, -1), (-1, 1)],  # 玉
            [(1, 0), (1, 1), (1, -1), (0, -1), (0, 1), (-1, 0), (-1, -1), (-1, 1)],  # 王
        ]
    
        return naru_pieces[piece_type - 1] if is_naru else pieces[piece_type - 1]

    def is_sliding_piece(self, piece_type, is_naru):
        """長距離移動する駒かどうかを判定"""
        if is_naru:
            # 成り駒の場合：竜馬、竜王は長距離移動
            return piece_type in [6, 7]  # 角、飛車の成り
        else:
            # 通常駒の場合：香車、角、飛車は長距離移動
            return piece_type in [2, 6, 7]  # 香車、角、飛車

    def get_ables(self, pos):
        """指定位置の駒が移動可能な位置を取得（飛び越え防止）"""
        ables = []
        piece_type, piece_whose, piece_naru = self.state[pos[0], pos[1]]
        
        if piece_whose != self.turn:
            return ables
        
        directions = self.get_selectable(piece_type, piece_naru)
        direction_multiplier = -1 if self.turn == 1 else 1
        
        for direction in directions:
            dy, dx = direction
            dy *= direction_multiplier
            dx *= direction_multiplier
            
            # 桂馬の特殊処理
            if piece_type == 3 and not piece_naru:  # 桂馬
                move_to = pos + np.array([dy, dx])
                if (0 <= move_to[0] < 9 and 0 <= move_to[1] < 9):
                    target_piece = self.state[move_to[0], move_to[1]]
                    if target_piece[0] == 0 or target_piece[1] != self.turn:
                        ables.append(move_to)
                continue
        
            # 長距離移動する駒かどうかチェック
            is_sliding = self.is_sliding_piece(piece_type, piece_naru)
            max_distance = 8 if is_sliding else 1
            
            # 各方向について1マスずつチェック
            for distance in range(1, max_distance + 1):
                move_to = pos + np.array([dy * distance, dx * distance])
                
                # 盤面外チェック
                if not (0 <= move_to[0] < 9 and 0 <= move_to[1] < 9):
                    break
                    
                target_piece = self.state[move_to[0], move_to[1]]
                
                # 空きマスの場合：移動可能、探索継続
                if target_piece[0] == 0:
                    ables.append(move_to)
                    continue
                
                # 相手の駒の場合：取れるが、それ以上は進めない
                elif target_piece[1] != self.turn:
                    ables.append(move_to)
                    break
                
                # 自分の駒の場合：移動不可、探索停止
                else:
                    break
                
        return ables

    def is_nifu(self, piece_type, pos):
        """二歩チェック"""
        if piece_type != 1:  # 歩でなければOK
            return False
        
        # 同じ列に自分の歩があるかチェック
        for row in range(9):
            if (self.state[row, pos[1]] == [1, self.turn, 0]).all():
                return True
        return False

    def get_legal_action(self, turn):
        """合法手を生成"""
        legal_actions = []
        
        # 盤上の駒の移動
        for y in range(9):
            for x in range(9):
                piece = self.state[y, x]
                if piece[1] == turn:
                    ables = self.get_ables(np.array([y, x]))
                    for to_pos in ables:
                        to_y, to_x = to_pos
                        
                        # 通常の移動
                        action = {
                            "motigoma_use": 0,
                            "koma_from": np.array([y, x]),
                            "koma_to": np.array([to_y, to_x]),
                            "motigoma_type": 0,
                            "motigoma_pos": np.array([0, 0]),
                            "wanna_naru": 0
                        }
                        legal_actions.append(action)
                        
                        # 成りの選択肢
                        if self.can_promote(piece[0]) and piece[2] == 0:
                            # 成れる条件をチェック
                            if ((turn == 1 and (y <= 2 or to_y <= 2)) or 
                                (turn == -1 and (y >= 6 or to_y >= 6))):
                                action_naru = action.copy()
                                action_naru["wanna_naru"] = 1
                                legal_actions.append(action_naru)

        # 持ち駒の配置
        player_idx = self.get_player_index(turn)
        for piece_type in range(1, 10):
            if self.motigoma[player_idx][piece_type] > 0:
                for y in range(9):
                    for x in range(9):
                        if self.state[y, x][0] == 0:  # 空きマス
                            # 二歩チェック
                            if self.is_nifu(piece_type, [y, x]):
                                continue
                                
                            action = {
                                "motigoma_use": 1,
                                "koma_from": np.array([0, 0]),
                                "koma_to": np.array([y, x]),
                                "motigoma_type": piece_type,
                                "motigoma_pos": np.array([y, x]),
                                "wanna_naru": 0
                            }
                            legal_actions.append(action)
        
        return legal_actions

    def put_koma(self, koma_from, koma_to, naru=0):
        """駒を移動"""
        # 境界チェック（修正済み）
        if not (0 <= koma_from[0] <= 8 and 0 <= koma_from[1] <= 8 and 
                0 <= koma_to[0] <= 8 and 0 <= koma_to[1] <= 8):
            return False

        piece_type, piece_whose, piece_naru = self.state[koma_from[0], koma_from[1]]
        
        if piece_whose != self.turn:
            return False

        ables = self.get_ables(koma_from)
        
        # 移動可能位置にあるかチェック（修正済み）
        move_valid = False
        for able_pos in ables:
            if np.array_equal(able_pos, koma_to):
                move_valid = True
                break
                
        if not move_valid:
            return False
            
        target_piece = self.state[koma_to[0], koma_to[1]]
        
        # 自分の駒がある場所には移動できない
        if target_piece[1] == self.turn:
            return False
            
        # 相手の駒を取る
        if target_piece[1] == -self.turn:
            player_idx = self.get_player_index(self.turn)
            captured_type = target_piece[0]
            # 成り駒は元の駒に戻す
            if target_piece[2] == 1:
                captured_type = captured_type
            self.motigoma[player_idx][captured_type] += 1

        # 駒を移動
        self.state[koma_to[0], koma_to[1]] = self.state[koma_from[0], koma_from[1]].copy()
        self.state[koma_from[0], koma_from[1]] = np.array([0, 0, 0])

        # 成り処理
        if naru == 1 and self.can_promote(piece_type) and piece_naru == 0:
            promotion_zone = (koma_to[0] <= 2 and self.turn == 1) or (koma_to[0] >= 6 and self.turn == -1)
            if promotion_zone:
                self.state[koma_to[0], koma_to[1]][2] = 1

        return True

    def put_motigoma(self, piece_type, pos):
        """持ち駒を配置"""
        if self.state[pos[0], pos[1]][0] != 0:
            return False
            
        if self.is_nifu(piece_type, pos):
            return False
            
        player_idx = self.get_player_index(self.turn)
        if self.motigoma[player_idx][piece_type] <= 0:
            return False
            
        self.state[pos[0], pos[1]] = np.array([piece_type, self.turn, 0])
        self.motigoma[player_idx][piece_type] -= 1
        return True

    def step(self, motigoma_use=False, koma_from=None, koma_to=None, 
             motigoma_type=None, motigoma_pos=None, wanna_naru=0):
        """一手進める"""
        if motigoma_use:
            result = self.put_motigoma(motigoma_type, motigoma_pos)
        else:
            result = self.put_koma(koma_from=koma_from, koma_to=koma_to, naru=wanna_naru)
        #print('result:' + str(result))
        if result:
            self.turn *= -1
            self.turn_count += 1
            
        done, winner_info = self.check_win()
        winner = None
        if done and isinstance(winner_info, list):
            if winner_info[0] == 1:
                winner = 0  # 先手勝利
            elif winner_info[1] == 1:
                winner = 1  # 後手勝利
            else:
                winner = -1  # 引き分け
                
        return self.state, self.motigoma, self.turn, result, done, winner

    def check_win(self):
        """勝敗判定"""
        # 玉と王の存在チェック
        gyoku_exists = False
        ou_exists = False
        
        for row in self.state:
            for piece in row:
                if piece[0] == 8:  # 玉
                    gyoku_exists = True
                elif piece[0] == 9:  # 王
                    ou_exists = True
                    
        if not gyoku_exists:
            return True, [0, 1]  # 王側勝利
        if not ou_exists:
            return True, [1, 0]  # 玉側勝利

        # 千日手判定
        board_hash = hash(str(self.get_learning_data()))
        self.position_history.append(board_hash)
        if self.position_history.count(board_hash) >= 4:
            return True, [0, 0]  # 引き分け

        # 最大手数チェック
        if self.turn_count >= self.max_turn_count:
            return True, [0, 0]  # 引き分け

        return False, [gyoku_exists, ou_exists]

    def get_learning_data(self):
        """学習用データを生成"""
        data_1 = np.zeros((10, 9))
        data_2 = np.zeros((10, 9))

        for i, row in enumerate(self.state):
            for j, piece in enumerate(row):
                piece_value = piece[0] * (piece[2] * 9 + 1) if piece[0] != 0 else 0
                if piece[1] == 1:
                    data_1[i, j] = piece_value
                elif piece[1] == -1:
                    data_2[i, j] = piece_value

        # 持ち駒情報を追加
        for k, num in enumerate(self.motigoma[0].values()):
            if k < 9:
                data_1[9, k] = num

        for k, num in enumerate(self.motigoma[1].values()):
            if k < 9:
                data_2[9, k] = num

        return np.array([data_1, data_2]) if self.turn == 1 else np.array([data_2, data_1])

    def act_to_scalar(self, act):
        """行動を数値に変換"""
        if isinstance(act, dict):
            motigoma_use = act["motigoma_use"]
            koma_from = act["koma_from"]
            koma_to = act["koma_to"]
            motigoma_type = act["motigoma_type"]
            motigoma_pos = act["motigoma_pos"]
            wanna_naru = act["wanna_naru"]
        else:
            motigoma_use, koma_from, koma_to, motigoma_type, motigoma_pos, wanna_naru = act

        if motigoma_use == 0:  # 通常の移動
            koma_from_pos = koma_from[0] * 9 + koma_from[1]
            koma_to_pos = koma_to[0] * 9 + koma_to[1]
            score = (koma_from_pos * 81 + koma_to_pos) * 2 + wanna_naru
        else:  # 持ち駒打ち
            motigoma_pos_flatten = motigoma_pos[0] * 9 + motigoma_pos[1]
            score = 13122 + (motigoma_type - 1) * 81 + motigoma_pos_flatten

        return int(score)

    def scalar_to_act(self, scalar):
        """数値を行動に変換"""
        if scalar <= 13121:
            # 通常の移動手
            wanna_naru = scalar % 2
            move_index = scalar // 2
            koma_from_index = move_index // 81
            koma_to_index = move_index % 81

            koma_from = np.array([koma_from_index // 9, koma_from_index % 9])
            koma_to = np.array([koma_to_index // 9, koma_to_index % 9])

            return (0, koma_from, koma_to, 0, np.array([0, 0]), wanna_naru)
        else:
            # 持ち駒打ち
            idx = scalar - 13122
            motigoma_type = idx // 81 + 1
            motigoma_pos_flatten = idx % 81
            motigoma_pos = np.array([motigoma_pos_flatten // 9, motigoma_pos_flatten % 9])

            return (1, np.array([0, 0]), motigoma_pos, motigoma_type, motigoma_pos, 0)

    def copy(self):
        """ゲーム状態のコピーを作成"""
        return copy.deepcopy(self)



random.seed(1234)
torch.manual_seed(1234)

def softmax(x):
    x = np.array(x)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.in_dim = 2
        self.hid_dim = 128
        self.board_h = 10
        self.board_w = 9
        self.flat_dim = self.hid_dim * self.board_h * self.board_w

        self.cnn1 = nn.Conv2d(self.in_dim, self.hid_dim, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=0.3)

        self.policy_head = nn.Linear(self.flat_dim, 13689)
        self.value_head = nn.Linear(self.flat_dim, 1)

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = self.dropout(x)
        x = F.relu(self.cnn2(x))
        x = self.dropout(x)
        x = F.relu(self.cnn3(x))
        x = self.dropout(x)

        x_flat = x.view(x.size(0), -1)
        x_flat = self.dropout(x_flat)

        policy = self.policy_head(x_flat)
        value = torch.tanh(self.value_head(x_flat))

        return policy, value

class TreeNode:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = prior_p

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda act_node:
                   act_node[1].Q + c_puct * act_node[1].P *
                   (self.N ** 0.5 / (1 + act_node[1].N)))

    def update(self, leaf_value):
        self.N += 1
        self.W += leaf_value
        self.Q = self.W / self.N

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

class MCTS:
    def __init__(self, policy_value_fn, c_puct=1.4, n_simulations=800):
        self.root = TreeNode(None, 1.0)
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def playout(self, env):
        """修正：ゲーム終了時に探索を打ち切る"""
        env_copy = env.copy()
        node = self.root

        while node.children:
            # 勝敗がついていたら探索打ち切り
            done, _ = env_copy.check_win()
            if done:
                return

            action, node = node.select(self.c_puct)
            act = env_copy.scalar_to_act(action)
            env_copy.step(*act)

        # 探索展開前にも勝敗チェック
        done, who = env_copy.check_win()
        if done:
            if who[0] == 1:# ゲーム終了時の価値を 0（引き分け）または適切に設定
                leaf_value = 1
            elif who[1] == 1:
                leaf_value = -1
            else:
                leaf_value = 0
            node.update_recursive(leaf_value)
            return

        action_probs, leaf_value = self.policy_value_fn(env_copy)
        node.expand(action_probs)
        node.update_recursive(leaf_value)


    def get_move_probs(self, env, temp=1e-3):
        for _ in range(self.n_simulations):
            done, _ = env.check_win()
            if done:
                break
            self.playout(env)

        if not self.root.children:
            #print('baka')
            return [], []
        acts, visits = zip(*[(act, node.N) for act, node in self.root.children.items()])
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        
        #print(acts)
        return acts, act_probs

class Trainer:
    def __init__(self, policy_value_net, env_cls, buffer_size=4096, batch_size=512, lr=1e-3):
        self.policy_value_net = policy_value_net
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr, weight_decay=1e-4)
        self.env_cls = env_cls
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def policy_value_fn(self, env):
        """修正済み：適切な合法手マスキング"""
        state = env.get_learning_data()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, value = self.policy_value_net(state_tensor)
            
        # 合法手のマスキング
        legal_actions = env.get_legal_action(env.turn)
        legal_action_ids = [env.act_to_scalar(act) for act in legal_actions]
        
        # 非合法手を大きな負の値でマスク
        masked_logits = logits.clone()
        mask = torch.full_like(masked_logits, -1e9)
        mask[0, legal_action_ids] = 0
        masked_logits = masked_logits + mask
        
        probs = F.softmax(masked_logits, dim=1).squeeze(0).cpu().numpy()
        action_probs = [(aid, probs[aid]) for aid in legal_action_ids]
        
        return action_probs, value.item()

    def self_play(self, n_games=32, mcts_simulations=400):
        """セルフプレイでデータ生成"""
        mcts = MCTS(self.policy_value_fn, n_simulations=mcts_simulations)

        for game_idx in range(n_games):
            env = self.env_cls()
            states, mcts_probs, current_players = [], [], []
            
            #print('????????')
            while True:
                done, winner_info = env.check_win()
                if done:
                    break
                    
                state = env.get_learning_data()
                #print(state)
                acts, probs = mcts.get_move_probs(env)
                if not acts:  # 合法手がない場合
                    
                    break
                    
                # 行動選択
                probs = np.array(probs)
                probs = probs / np.sum(probs)  # 正規化
                move = np.random.choice(acts, p=probs)
                act = env.scalar_to_act(move)
                
                # 状態保存
                states.append(state)
                prob_full = np.zeros(13689, dtype=np.float32)
                for a, p in zip(acts, probs):
                    prob_full[a] = p
                mcts_probs.append(prob_full)
                current_players.append(env.turn)
                
                # 行動実行
                env.step(*act)

            # 勝者決定
            done, winner_info = env.check_win()
            if isinstance(winner_info, list):
                if winner_info[0] == 1:
                    winner = 1  # 先手勝利
                elif winner_info[1] == 1:
                    winner = -1  # 後手勝利
                else:
                    winner = 0  # 引き分け
            else:
                winner = 0

            #print(winner)

            # データをバッファに追加
            for i in range(len(states)):
                z = winner if current_players[i] == 1 else -winner
                self.buffer.append((states[i], mcts_probs[i], z))
                
            if game_idx % 10 == 0:
                print(f"Self-play game {game_idx + 1}/{n_games} completed")

    def learn(self, epochs=5):
        """学習実行"""
        if len(self.buffer) < self.batch_size:
            return None

        total_loss = 0
        for epoch in range(epochs):
            mini_batch = random.sample(self.buffer, self.batch_size)

            state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s, _, _ in mini_batch]).to(device)
            mcts_prob_batch = torch.tensor(np.array([p for _, p, _ in mini_batch]), dtype=torch.float32).to(device)
            z_batch = torch.tensor(np.array([z for _, _, z in mini_batch]), dtype=torch.float32).view(-1, 1).to(device)

            logits, value = self.policy_value_net(state_batch)
            
            # 損失計算
            value_loss = F.mse_loss(value, z_batch)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -torch.mean(torch.sum(mcts_prob_batch * log_probs, dim=1))
            loss = value_loss + policy_loss

            # 最適化
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / epochs

# メイン実行部分
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"モデルを保存しました: {path}")

def load_model(model, path):
    print(path)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"モデルを読み込みました: {path}")
        return True
    return False

def main():
    # 初期化
    syogi = Syogi()
    agent = Agent().to(device)
    trainer = Trainer(policy_value_net=agent, env_cls=Syogi, buffer_size=8192, batch_size=256, lr=2e-4)
    test_env = Syogi()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    MODE = 'train'
    MAX_EPOCH = 100000

    if MODE == 'train':
        print('Training start...')
        
        # 既存モデルの読み込み試行
        load_model(agent, os.path.join(base_dir, "model.pth"))
        
        for i in range(MAX_EPOCH):
            print(f"\nEpoch {i + 1}/{MAX_EPOCH}")
            
            # セルフプレイ
            trainer.self_play(n_games=16, mcts_simulations=200)
            
            # 学習
            loss = trainer.learn(epochs=3)
            if loss is not None:
                print(f"Loss: {loss:.4f}")
            else:
                print(f'Buffer size: {len(trainer.buffer)} (minimum: {trainer.batch_size})')
            
            # モデル保存
            if (i + 1) % (MAX_EPOCH // 10) == 0:
                save_model(agent, os.path.join(base_dir, f"model_epoch_{i + 1}.pth"))
            
            # 定期的なテスト
            if (i + 1) % (MAX_EPOCH // 10) == 0:
                print("Testing model...")
                state = test_env.get_learning_data()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                
                agent.eval()
                with torch.no_grad():
                    logits, value = agent(state_tensor)
                    legal_actions = test_env.get_legal_action(test_env.turn)
                    if legal_actions:
                        legal_ids = [test_env.act_to_scalar(act) for act in legal_actions]
                        best_action_id = legal_ids[torch.argmax(logits[0, legal_ids]).item()]
                        best_action = test_env.scalar_to_act(best_action_id)
                        test_env.step(*best_action)
                        print(state)
                        print(f"Best action: {best_action}, Value: {value.item():.3f}")
                agent.train()
        
        save_model(agent, os.path.join(base_dir, "final_model.pth"))
        print("Training completed!")
        
    else:
        # テストモード
        if load_model(agent, os.path.join(base_dir, 'model_epoch_100.pth')):
            test_env = Syogi()
            for _ in range(30):
                state = test_env.get_learning_data()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                
                agent.eval()
                
                with torch.no_grad():
                    logits, value = agent(state_tensor)
                    legal_actions = test_env.get_legal_action(test_env.turn)
                    if legal_actions:
                        legal_ids = [test_env.act_to_scalar(act) for act in legal_actions]
                        best_action_id = legal_ids[torch.argmax(logits[0, legal_ids]).item()]
                        best_action = test_env.scalar_to_act(best_action_id)
                        print(f"Recommended action: {best_action}")
                        print(f"Position value: {value.item():.3f}")
                        print(state)
                        test_env.step(*best_action)

if __name__ == "__main__":
    main()
