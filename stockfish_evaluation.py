import time
import chess
import chess.engine
import chess.pgn

DEFAULT_DEPTH = 10


class StockfishEvaluator:
    def __init__(self, stockfish_path="/usr/bin/stockfish", depth=DEFAULT_DEPTH):
        self.depth = depth
        self.limit = chess.engine.Limit(depth=depth)
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def _score_cp(self, board):
        if board.is_game_over():
            return None
        info = self.engine.analyse(board, self.limit, game=object())
        return info["score"].relative.score(mate_score=10000)

    def _best_moves(self, board, k):
        """Stockfish's k best moves, best first."""
        if board.is_game_over():
            return []
        infos = self.engine.analyse(board, self.limit, multipv=k, game=object())
        return [i["pv"][0] for i in infos if i.get("pv")]

    def _centipawn_loss(self, board, move, score_before):
        """How much value the model's move gave away."""
        board.push(move)
        try:
            score_after_opponent_view = self._score_cp(board)
        finally:
            board.pop() 

        if score_after_opponent_view is None:
            return 0
        score_after_our_view = -score_after_opponent_view
        return max(0, score_before - score_after_our_view)

    def evaluate(self, model, positions, top_k=3, progress_every=25, verbose=True):
        name = getattr(model, "name", type(model).__name__)
        restricted = getattr(model, "restricted_to_legal_moves", False)
        has_top_k = hasattr(model, "predict_top_k")

        n = 0
        no_prediction = 0
        legal = 0
        top1_hits = 0
        topk_hits = 0
        cp_losses = []

        if verbose:
            print(f"\nEvaluating: {name}")
            print(f"  {len(positions)} positions, Stockfish depth {self.depth}")

        start = time.time()

        for i, original in enumerate(positions, start=1):
            if original.is_game_over():
                continue

            board = original.copy()  
            n += 1

            move = model.predict(board)

            if move is None:
                no_prediction += 1
                continue

            if move not in board.legal_moves:
                continue

            legal += 1

            best = self._best_moves(board, top_k)
            if best:
                if move == best[0]:
                    top1_hits += 1
                if move in best:
                    topk_hits += 1

            score_before = self._score_cp(board)
            if score_before is not None:
                cp_losses.append(self._centipawn_loss(board, move, score_before))

            if verbose and i % progress_every == 0:
                elapsed = time.time() - start
                print(f"    ...{i}/{len(positions)} "
                      f"({i/elapsed:.1f} pos/sec, {elapsed:.0f}s)")

        elapsed = time.time() - start
        scored = len(cp_losses)

        results = {
            "name": name,
            "positions_attempted": n,
            "no_prediction": no_prediction,
            "legal_moves": legal,
            "legal_rate": legal / n if n else 0.0,
            "legal_rate_is_meaningful": not restricted,
            "top1_agreement": top1_hits / n if n else 0.0,
            "topk_agreement": topk_hits / n if n else 0.0,
            "top_k": top_k,
            "avg_centipawn_loss": sum(cp_losses) / scored if scored else None,
            "median_centipawn_loss": sorted(cp_losses)[scored // 2] if scored else None,
            "blunders_over_300cp": sum(1 for c in cp_losses if c > 300),
            "scored_positions": scored,
            "seconds": elapsed,
            "depth": self.depth,
            "model_has_top_k": has_top_k,
        }
        return results

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def positions_from_pgn(pgn_path, max_positions=200, skip_opening_moves=0):
    positions = []
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while len(positions) < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                if ply >= skip_opening_moves:
                    positions.append(board.copy())
                board.push(move)
                if len(positions) >= max_positions:
                    break
    return positions

def print_report(all_results):
    print("\n" + "=" * 78)
    print("EVALUATION RESULTS")
    print("=" * 78)

    header = f"{'Model':<32} {'Legal':>8} {'Top-1':>8} {'Top-3':>8} {'Avg CPL':>9}"
    print(header)
    print("-" * 78)

    for r in all_results:
        legal = f"{r['legal_rate']:.1%}"
        if not r["legal_rate_is_meaningful"]:
            legal += "*"
            footnote_needed = True
        cpl = f"{r['avg_centipawn_loss']:.0f}" if r["avg_centipawn_loss"] is not None else "n/a"
        print(f"{r['name'][:32]:<32} {legal:>8} {r['top1_agreement']:>7.1%} "
              f"{r['topk_agreement']:>7.1%} {cpl:>9}")

    print("-" * 78)
    print(f"\nStockfish depth: {all_results[0]['depth']}")
    print("\nDetail:")
    for r in all_results:
        print(f"\n  {r['name']}")
        print(f"    positions attempted   : {r['positions_attempted']}")
        print(f"    no prediction returned: {r['no_prediction']}")
        print(f"    median CPL            : {r['median_centipawn_loss']}")
        print(f"    blunders (>300cp)     : {r['blunders_over_300cp']} "
              f"of {r['scored_positions']} scored")
        print(f"    time                  : {r['seconds']:.0f}s")