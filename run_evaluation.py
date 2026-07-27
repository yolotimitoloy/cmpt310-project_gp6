import argparse
import warnings
import chess
 
warnings.filterwarnings("ignore")
 
import baseline_models
import stockfish_evaluation
from stockfish_evaluation import StockfishEvaluator, positions_from_pgn, print_report
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stockfish", default="/usr/bin/stockfish",
                   help="path to the Stockfish binary")
    p.add_argument("--pgn", default="verifSet/verif.pgn",
                   help="PGN to draw test positions from (HELD-OUT games only)")
    p.add_argument("--positions", type=int, default=30)
    p.add_argument("--skip-opening", type=int, default=0,
                   help="ignore first N plies per game; use ~10 for middlegame")
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--model", default="model.joblib")
    p.add_argument("--no-selftest", action="store_true")
    p.add_argument("--elo", type=int, default=1400,
                   help="Elo for the weakened-Stockfish mid reference")
    args = p.parse_args()
 
    baseline_models.STOCKFISH_PATH = args.stockfish
 
    print(f"Loading positions from {args.pgn} ...")
    positions = positions_from_pgn(
        args.pgn,
        max_positions=args.positions,
        skip_opening_moves=args.skip_opening,
    )
    print(f"  got {len(positions)} positions"
          + (f" (skipping first {args.skip_opening} plies per game)"
             if args.skip_opening else ""))
 
    if not positions:
        print("\nNo positions found. Check the PGN path.")
        return
 
    models = []
 
    if not args.no_selftest:
        models.append(baseline_models.StockfishMover(
            path=args.stockfish, depth=args.depth, multipv=3))
 
    try:
        from adapters import MLPAdapter
        models.append(MLPAdapter(args.model))
    except FileNotFoundError:
        print(f"\n  {args.model} not found -- skipping the MLP.")
    except Exception as e:
        print(f"\n  Could not load MLP: {type(e).__name__}: {e}")
 
    try:
        models.append(baseline_models.WeakStockfishMover(
            path=args.stockfish, depth=args.depth, elo=args.elo, multipv=3))
    except Exception as e:
        print(f"\n  Could not create weakened Stockfish: {e}")
 
    models.append(baseline_models.RandomMover())
 
    results = []
    try:
        with StockfishEvaluator(args.stockfish, depth=args.depth) as evaluator:
            for model in models:
                results.append(evaluator.evaluate(model, positions))
    finally:
        for model in models:
            model.close()
 
    print_report(results)
 
    if not args.no_selftest and results:
        sf = results[0]
        print("\n" + "=" * 78)
        if sf["top1_agreement"] > 0.95 and (sf["avg_centipawn_loss"] or 0) < 15:
            print("SELF-TEST PASSED -- evaluator agrees with Stockfish about itself.")
            print("The other numbers in this run can be trusted.")
        else:
            print("SELF-TEST FAILED -- the evaluator has a bug, OR the depth used")
            print("by StockfishMover does not match the evaluator's depth.")
            print(f"  got top-1 {sf['top1_agreement']:.1%}, "
                  f"avg CPL {sf['avg_centipawn_loss']}")
            print("Do not report the other numbers until this reads ~100% / ~0.")
        print("=" * 78)
 
 
if __name__ == "__main__":
    main()