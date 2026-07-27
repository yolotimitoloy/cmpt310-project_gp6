"""
Usage: python train_tim.py pgnFiles --max-positions 100000
"""
import argparse
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from preprocessing import get_processed_data


def main():

    # 1. Command Line Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn_folder", help="Folder containing .pgn files")
    parser.add_argument("--max-positions", type=int, default=100_000)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--checkpoint-out", default="model.joblib")
    args = parser.parse_args()

    # 2. Data Loading & Preprocessing: X represents the board states (features), y represents the moves played (labels)
    X, y = get_processed_data(args.pgn_folder, max_positions=max_positions)

    # 3. Data Splitting: 98% for training, 2% for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.02, random_state=0)

    #4. Model Configuration: Multi-Layer Perceptron (Neural Network) classifier
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

    # 5. Fit the model to the training data
    print("\nTraining MLPClassifier ...")
    clf.fit(X_train, y_train)

    # 6. Evaluation & Export
    print(f"train_move_acc={clf.score(X_train, y_train):.3f}  "
          f"val_move_acc={clf.score(X_val, y_val):.3f}")

    # 7. Score the model to see how accurately it predicts moves on both seen and unseen data
    joblib.dump(clf, args.checkpoint_out)
    print(f"Saved model to {args.checkpoint_out}")


if __name__ == "__main__":
    main()