"""
Usage: python predict_tim.py chess_model_preprocessing.joblib --moves "e4 c5 Nf3"
"""
import argparse
import joblib
import chess

from preprocessing import board_to_one_hot, move_tuple_to_label


def suggest_moves(clf, board, top_k=1):
    one_hot = board_to_one_hot(board)
    x = one_hot.reshape(1, -1).astype("float32")

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    probs = clf.predict_proba(x)[0]
    class_to_prob = dict(zip(clf.classes_, probs))

    scored = []
    for move in legal_moves:
        label = move_tuple_to_label(move.from_square, move.to_square, move.promotion)
        scored.append((move, class_to_prob.get(label, 0.0)))

    scored.sort(key=lambda pair: -pair[1])
    return scored[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--fen", default=None)
    parser.add_argument("--moves", default=None)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    clf = joblib.load(args.checkpoint)

    board = chess.Board()
    if args.fen:
        board = chess.Board(args.fen)
    elif args.moves:
        for san in args.moves.split():
            board.push_san(san)

    scored = suggest_moves(clf, board, top_k=args.top_k)

    print(f"\nPosition (side to move: {'White' if board.turn else 'Black'}):")
    print(board)
    print("\nTop move suggestions:")
    if not scored or all(p == 0.0 for _, p in scored):
        print("  (model assigned ~0 probability to every legal move here -- needs more training data)")
    for move, prob in scored:
        print(f"  {board.san(move):8s}  {prob:.1%}")


if __name__ == "__main__":
    main()
