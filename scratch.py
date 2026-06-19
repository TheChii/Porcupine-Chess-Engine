import chess

fen = "nqrbknbr/ppppp1pp/5p2/8/6P1/8/PPPPPP1P/RKNNBBQR w KQkq - 0 2"
board = chess.Board(fen, chess960=True)
print(board.fen())
