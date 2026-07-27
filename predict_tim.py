"""
Usage: python predict_tim.py chess_model_preprocessing.joblib --moves "e4 c5 Nf3"
"""
import argparse
import joblib
import chess

from preprocessing import board_to_one_hot, move_tuple_to_label


def suggest_moves(clf, board, top_k=1):
    # Convert the 2D chess board into a flat 1D array of 0s and 1s
    one_hot = board_to_one_hot(board)
    x = one_hot.reshape(1, -1).astype("float32")

    # 1. Get all moves that are allowed by the rules of chess
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return [] # Game is over (checkmate or stalemate)

    # 2. Ask the neural network for the probability of all possible moves: list of percentages
    probs = clf.predict_proba(x)[0]
    class_to_prob = dict(zip(clf.classes_, probs))

    # 3. Filter and score only the legal moves
    scored = []
    for move in legal_moves:
        label = move_tuple_to_label(move.from_square, move.to_square, move.promotion)
        scored.append((move, class_to_prob.get(label, 0.0)))

    # 4. Sort the moves so the highest probability is at the top
    scored.sort(key=lambda pair: -pair[1])

    # 5. Return only the top requested number of moves
    return scored[:top_k]


def main():
    # 1. Command Line Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--fen", default=None)
    parser.add_argument("--moves", default=None)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    # 2. Load the pre-trained neural network
    clf = joblib.load(args.checkpoint)

    board = chess.Board()
    if args.fen:
        board = chess.Board(args.fen)
    elif args.moves:
        for san in args.moves.split():
            board.push_san(san)

    # 3. Calculate the best moves for this specific board state
    scored = suggest_moves(clf, board, top_k=args.top_k)

    # 4. Display Results
    print(f"\nPosition (side to move: {'White' if board.turn else 'Black'}):")
    print(board)
    print("\nTop move suggestions:")
    # Edge case: the model has no idea what to do because it hasn't seen data similar to this
    if not scored or all(p == 0.0 for _, p in scored):
        print("  (model assigned ~0 probability to every legal move here -- needs more training data)")
    # Print the chosen moves and their probabilities
    for move, prob in scored:
        print(f"  {board.san(move):8s}  {prob:.1%}")


if __name__ == "__main__":
    main()
