"""
Converts (board, move) pairs from board_parser.py into arrays ready for
sklearn: X of shape (N, 13*8*8) and y of shape (N,) integer move labels.

Two fixes vs. the original version:

1. SIDE-TO-MOVE PLANE ADDED (plane 12). The original only encoded piece
   positions -- two positions with identical pieces but different sides to
   move were indistinguishable to the model, even though "best move" is
   meaningless without knowing whose turn it is. Plane 12 is filled with all
   1s if it's White to move, all 0s if Black (a constant plane is a
   standard, if slightly wasteful-looking, way to inject a scalar fact into
   a conv/flattened representation).

2. PROMOTION IS NOW PART OF THE LABEL. The original label was
   (from_square, to_square, piece_type_of_mover) -- promoting a pawn to a
   queen vs. underpromoting to a knight produced the IDENTICAL label, since
   only the moving piece's *pre-move* type was recorded. Underpromotions
   became unrecoverable/ambiguous. Labels are now encoded as a single
   integer via move_tuple_to_label(), which folds in the promotion choice.

Label space: 64 (from) x 64 (to) x 5 (promotion: none/N/B/R/Q) = 20,480
possible labels. This is a superset -- most (from, to, promotion)
combinations are never legal in any position (e.g. promotion on a non-pawn
move) -- but that's fine for a classifier target; sklearn just never sees
those classes populated. If you want a denser, smaller action space (4,672
classes with no invalid combinations), see move_encoding.py from the
CNN pipeline, which uses the AlphaZero 8x8x73 scheme instead.
"""
import time
import numpy as np
import chess
import board_parser

NUM_PLANES = 13  # 12 piece planes + 1 side-to-move plane

_PIECE_TO_INT = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}

_PROMOTION_TO_INT = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
_INT_TO_PROMOTION = {v: k for k, v in _PROMOTION_TO_INT.items()}


def piece_to_int(piece_symbol):
    return _PIECE_TO_INT[piece_symbol]


def get_one_hot():
    return np.zeros((NUM_PLANES, 8, 8), dtype=bool)


def move_tuple_to_label(from_square, to_square, promotion):
    """promotion: None, chess.KNIGHT, chess.BISHOP, chess.ROOK, or chess.QUEEN"""
    promo_int = _PROMOTION_TO_INT[promotion]
    return from_square * (64 * 5) + to_square * 5 + promo_int


def label_to_move(label, board):
    """Decodes a label back into a chess.Move, legal-move-checkable against board."""
    from_square, remainder = divmod(label, 64 * 5)
    to_square, promo_int = divmod(remainder, 5)
    promotion = _INT_TO_PROMOTION[promo_int]
    return chess.Move(from_square, to_square, promotion=promotion)


def board_to_one_hot(board):
    one_hot = get_one_hot()
    for square, piece in board.piece_map().items():
        layer = piece_to_int(piece.symbol())
        x_ind = square % 8
        y_ind = square // 8
        one_hot[layer][x_ind][y_ind] = True

    if board.turn == chess.WHITE:
        one_hot[12, :, :] = True
    # else leave as all-False for Black to move

    return one_hot


def get_processed_data(folder_name, max_positions=None):
    """
    Returns (X, y) as numpy arrays ready for sklearn:
      X: shape (N, 13*8*8), dtype float32 (flattened, since sklearn wants 2D input)
      y: shape (N,), dtype int64 (labels from move_tuple_to_label)

    max_positions: if set, parsing stops as soon as this many positions have
    been collected, instead of parsing the whole file/folder and discarding
    the rest. Important for large files -- see the note in board_parser.py.
    """
    data = board_parser.get_training_data_from_folder(folder_name, max_positions=max_positions)

    print(f"Encoding {len(data)} positions ...")
    encode_start = time.time()
    encodings = []
    labels = []
    progress_every = max(1, len(data) // 20)  # ~20 progress lines
    for i, (board, move) in enumerate(data, start=1):
        one_hot = board_to_one_hot(board)
        label = move_tuple_to_label(move.from_square, move.to_square, move.promotion)
        encodings.append(one_hot)
        labels.append(label)
        if i % progress_every == 0 or i == len(data):
            elapsed = time.time() - encode_start
            rate = i / elapsed
            print(f"  ...{i}/{len(data)} positions encoded ({rate:.0f}/sec, {elapsed:.1f}s elapsed)")

    X = np.stack(encodings).reshape(len(encodings), -1).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    print(f"Encoding done in {time.time()-encode_start:.1f}s. X shape: {X.shape}")
    return X, y


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "pgnFiles"
    X, y = get_processed_data(folder)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Distinct move labels: {len(np.unique(y))}")