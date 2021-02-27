from cshogi import *
from cshogi import CSA
import numpy as np
import os
import glob
import lzma
import math
import argparse


def convert_eval(p):
    # dlshogiが使っている評価値を勝率に変換するときの定数
    a = 0.0013226

    e = np.log(max(p, 1e-3) / max(1 - p, 1e-3)) / a
    # 通常の評価値はScoreMaxEvaluate=30000が上限のはず
    e = np.clip(np.round(e), a_min=-30000, a_max=30000)
    return e


parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('out_dir')
args = parser.parse_args()

csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa*'), recursive=True)
os.makedirs(args.out_dir, exist_ok=True)

hcpes = np.zeros(10000*512, HuffmanCodedPosAndEval)
# 古いバージョンの棋譜だと評価値の情報がない
# dlshogiのScoreNone=32602 で初期化する
hcpes['eval'] = 32602

board = Board()
kif_num = 0
position_num = 0
for filepath in csa_file_list:
    print(filepath)
    p = 0
    if filepath.endswith('.xz'):
        file = lzma.open(filepath, 'rt')
        filepath = filepath[:-3]
    else:
        file = filepath
    for kif in CSA.Parser.parse_file(file):
        if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI', '%HIKIWAKE', '%CHUDAN') or len(kif.moves) <= 30:
            continue
        kif_num += 1
        board.set_sfen(kif.sfen)
        # 30手までで最善手以外が指された手番を見つける
        start = -1
        for i, (move, comment) in enumerate(zip(kif.moves, kif.comments)):
            comments = comment.decode('ascii').split(',')
            if comments[0].startswith('v='):
                candidates = comments[1:]
            else:
                candidates = comments
            if board.move_from_csa(candidates[1]) != move:
                start = i
            if i >= 29:
                break
            board.push(move)

        # 最後に最善手以外が指された局面から開始する
        board.set_sfen(kif.sfen)
        start_p = p
        for i, (move, comment) in enumerate(zip(kif.moves, kif.comments)):
            if i <= start:
                board.push(move)
                continue
            # 5手詰みチェック
            if board.mate_move(5) != 0:
                if kif.win != board.turn + 1:
                    # 詰みを見逃して逆転したゲームの結果を修正
                    hcpes[start_p:p]['gameResult'] = board.turn + 1
                break
            hcpe = hcpes[p]
            board.to_hcp(hcpe['hcp'])
            hcpe['bestMove16'] = move16(move)
            hcpe['gameResult'] = kif.win
            comments = comment.decode('ascii').split(',')
            if comments[0].startswith('v='):
                hcpe['eval'] = convert_eval(float(comments[0][2:]))
            p += 1
            board.push(move)

    hcpes[:p].tofile(os.path.join(args.out_dir, os.path.splitext(os.path.basename(filepath))[0] + '.hcpe'))
    position_num += p

print('kif_num', kif_num)
print('position_num', position_num)
