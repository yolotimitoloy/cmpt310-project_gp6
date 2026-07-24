"""
Usage: python train_tim.py pgnFiles --max-positions 100000
"""
import argparse
import time
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from preprocessing import get_processed_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn_folder", help="Folder containing .pgn files")
    parser.add_argument("--max-positions", type=int, default=100_000)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--checkpoint-out", default="model.joblib")
    args = parser.parse_args()

    max_positions = args.max_positions if args.max_positions else None

    print(f"Parsing PGNs and building features from '{args.pgn_folder}' ...")
    start = time.time()
    X, y = get_processed_data(args.pgn_folder, max_positions=max_positions)
    print(f"Done in {time.time()-start:.1f}s. X={X.shape}, y={y.shape}, "
          f"{len(np.unique(y))} distinct move labels present")

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
    start = time.time()
    clf.fit(X_train, y_train)
    print(f"\nTraining took {(time.time()-start)/60:.1f} min, {clf.n_iter_} iterations")

    print(f"train_move_acc={clf.score(X_train, y_train):.3f}  "
          f"val_move_acc={clf.score(X_val, y_val):.3f}")

    joblib.dump(clf, args.checkpoint_out)
    print(f"Saved model to {args.checkpoint_out}")


if __name__ == "__main__":
    main()