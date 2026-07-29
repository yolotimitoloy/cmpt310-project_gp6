"""
Usage: python train_tim.py pgnFiles --max-positions 100000
"""
import argparse
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from preprocessing import get_processed_data
from board_parser import RESULT_ALIASES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn_folder", help="Folder containing .pgn files")
    parser.add_argument("--max-positions", type=int, default=100_000)
    parser.add_argument("--min-elo", type=int, default=None)
    parser.add_argument("--result", choices=["any", "white", "black", "draw"], default="any")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--checkpoint-out", default="model.joblib")
    args = parser.parse_args()

    results_allowed = RESULT_ALIASES[args.result]

    X, y = get_processed_data(
        args.pgn_folder,
        max_positions=args.max_positions,
        min_elo=args.min_elo,
        results_allowed=results_allowed,
    )

    if len(X) == 0:
        print("No positions matched your filters -- try loosening --min-elo or --result.")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.02, random_state=0)

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation="relu",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=args.max_iter,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.05,
        verbose=True,
        random_state=0,
    )

    print("\nTraining MLPClassifier ...")
    clf.fit(X_train, y_train)

    print(f"train_move_acc={clf.score(X_train, y_train):.3f}  "
          f"val_move_acc={clf.score(X_val, y_val):.3f}")

    joblib.dump(clf, args.checkpoint_out)
    print(f"Saved model to {args.checkpoint_out}")


if __name__ == "__main__":
    main()