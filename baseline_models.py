import random
import chess
import chess.engine
 
STOCKFISH_PATH = "/usr/bin/stockfish"
 
 
class RandomMover:
    """The floor. Uniform random choice among legal moves."""
 
    name = "Random (baseline floor)"
 
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
 
    def predict(self, board):
        legal = list(board.legal_moves)
        if not legal:
            return None
        return self.rng.choice(legal)
 
    def predict_top_k(self, board, k=3):
        legal = list(board.legal_moves)
        self.rng.shuffle(legal)
        return legal[:k]
 
    def close(self):
        pass
 
 
class StockfishMover:
    name = "Stockfish (baseline ceiling)"
 
    def __init__(self, path=STOCKFISH_PATH, depth=10, multipv=3):
        self.depth = depth
        self.multipv = multipv  
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
 
    def predict(self, board):
        if board.is_game_over():
            return None
        try:
            infos = self.engine.analyse(
                board, chess.engine.Limit(depth=self.depth),
                multipv=self.multipv, game=object(),  
            )
            for info in infos:
                if info.get("pv"):
                    return info["pv"][0]
            return None
        except Exception:
            return None
 
    def predict_top_k(self, board, k=3):
        if board.is_game_over():
            return []
        try:
            infos = self.engine.analyse(
                board, chess.engine.Limit(depth=self.depth), multipv=k, game=object()
            )
            return [info["pv"][0] for info in infos if info.get("pv")]
        except Exception:
            return []
 
    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass
 
    def __enter__(self):
        return self
 
    def __exit__(self, *exc):
        self.close()
 
 
class WeakStockfishMover(StockfishMover):
    def __init__(self, path=STOCKFISH_PATH, depth=10, elo=1400, multipv=3):
        super().__init__(path=path, depth=depth, multipv=multipv)
        self.elo = elo
        self.name = f"Stockfish ~{elo} Elo (baseline mid)"
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
 
 
if __name__ == "__main__":
    print("=" * 62)
    print("Baseline sanity check")
    print("=" * 62)
 
    board = chess.Board()
 
    rnd = RandomMover()
    print(f"\n{rnd.name}")
    print(f"  opening move: {board.san(rnd.predict(board))}")
    print(f"  top 3:        {[board.san(m) for m in rnd.predict_top_k(board, 3)]}")
 
    try:
        with StockfishMover() as sf:
            print(f"\n{sf.name}")
            print(f"  opening move: {board.san(sf.predict(board))}")
            print(f"  top 3:        {[board.san(m) for m in sf.predict_top_k(board, 3)]}")
    except Exception as e:
        print(f"\n  Could not start Stockfish at {STOCKFISH_PATH}")
        print(f"  {type(e).__name__}: {e}")
        print("  Fix STOCKFISH_PATH at the top of this file.")