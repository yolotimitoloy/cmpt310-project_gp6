import chess


# ================================ Tim's MLP ================================


class MLPAdapter:
    restricted_to_legal_moves = True  
    
    def __init__(self, model_path="model.joblib", name=None):
        import joblib
        import predict_tim

        self._suggest = predict_tim.suggest_moves
        self.clf = joblib.load(model_path)
        self.name = name or f"MLP ({model_path})"

    def predict(self, board):
        try:
            scored = self._suggest(self.clf, board, top_k=1)
            if not scored:
                return None
            move, probability = scored[0]

            if probability <= 0.0:
                return None
            return move
        except Exception:
            return None

    def predict_top_k(self, board, k=3):
        try:
            scored = self._suggest(self.clf, board, top_k=k)
            return [move for move, probability in scored if probability > 0.0]
        except Exception:
            return []

    def close(self):
        pass


# =========================== KNN ============================


class KNNAdapter:
    restricted_to_legal_moves = False

    def __init__(self, knn, x, y, encode_fn, name="KNN"):
        """
        knn       : fitted KNeighborsClassifier
        x, y      : the training arrays predict_move needs for exact matching
        encode_fn : board -> flat numpy feature vector (same encoding used
                    to train, or predictions will be garbage)
        """
        import machine_learning

        self._predict_move = machine_learning.predict_move
        self.knn = knn
        self.x = x
        self.y = y
        self.encode = encode_fn
        self.name = name

    def predict(self, board):
        try:
            features = self.encode(board)
            label = self._predict_move(features, self.x, self.y, self.knn)
            return self._label_to_move(label, board)
        except Exception:
            return None

    def predict_top_k(self, board, k=3):
        move = self.predict(board)
        return [] if move is None else [move]

    @staticmethod
    def _label_to_move(label, board):
        """Turns (from_square, to_square, piece_type) into a chess.Move."""
        if label is None:
            return None

        from_square = int(label[0])
        to_square = int(label[1])

        promotion = None
        piece = board.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.PAWN:
            if chess.square_rank(to_square) in (0, 7):
                promotion = chess.QUEEN

        return chess.Move(from_square, to_square, promotion=promotion)

    def close(self):
        pass


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 62)
    print("Adapter smoke test")
    print("=" * 62)

    try:
        model = MLPAdapter("model.joblib")
        print(f"\nLoaded: {model.name}")
        print(f"  expects {model.clf.n_features_in_} features, "
              f"{len(model.clf.classes_)} output classes")

        for label, moves in [
            ("opening", []),
            ("after 1.e4", ["e4"]),
            ("rook endgame", None),
        ]:
            if moves is None:
                board = chess.Board("8/8/8/4k3/8/8/4K3/R7 w - - 0 1")
            else:
                board = chess.Board()
                for san in moves:
                    board.push_san(san)

            move = model.predict(board)
            top3 = model.predict_top_k(board, 3)
            print(f"\n  {label}:")
            print(f"    predict     -> {board.san(move) if move else None}")
            print(f"    top 3       -> {[board.san(m) for m in top3]}")
            print(f"    legal?      -> {move in board.legal_moves if move else 'n/a'}")

    except FileNotFoundError:
        print("\n  model.joblib not found -- are you on the tim-stuff branch?")
    except Exception as e:
        print(f"\n  {type(e).__name__}: {e}")

    print("\n  KNNAdapter: no trained KNN exists yet (machine_learning.py")
    print("  still has the 'int k = 1;' syntax error), so nothing to test.")