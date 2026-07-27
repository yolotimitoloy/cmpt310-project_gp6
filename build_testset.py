import argparse
import hashlib
import io
import os
import chess.pgn


def game_fingerprint(game):
    moves = " ".join(move.uci() for move in game.mainline_moves())
    return hashlib.sha256(moves.encode()).hexdigest()


def collect_fingerprints(paths):
    """Fingerprints of every game in the training files, to exclude later."""
    seen = set()
    for path in paths:
        if not os.path.exists(path):
            print(f"  WARNING: training file not found, skipping: {path}")
            continue
        count = 0
        with open(path, encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                seen.add(game_fingerprint(game))
                count += 1
        print(f"  {path}: {count} training games fingerprinted")
    return seen


def elo_ok(game, min_elo, max_elo):
    """Both players must fall inside the rating window."""
    if min_elo is None and max_elo is None:
        return True
    try:
        white = int(game.headers.get("WhiteElo", 0))
        black = int(game.headers.get("BlackElo", 0))
    except ValueError:
        return False  
    if white == 0 or black == 0:
        return False
    if min_elo is not None and (white < min_elo or black < min_elo):
        return False
    if max_elo is not None and (white > max_elo or black > max_elo):
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="PGN to draw test games from (decompressed)")
    p.add_argument("--out", default="verifSet/heldout.pgn")
    p.add_argument("--games", type=int, default=300,
                   help="how many games to write")
    p.add_argument("--training", nargs="*", default=[],
                   help="training PGNs to exclude by fingerprint")
    p.add_argument("--min-elo", type=int, default=None)
    p.add_argument("--max-elo", type=int, default=None)
    p.add_argument("--min-moves", type=int, default=20,
                   help="skip very short games (resignations, timeouts)")
    args = p.parse_args()

    print("Fingerprinting training data ...")
    training_ids = collect_fingerprints(args.training)
    print(f"  {len(training_ids)} unique training games total\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    kept = 0
    scanned = 0
    rejected_overlap = 0
    rejected_elo = 0
    rejected_short = 0

    exporter = chess.pgn.FileExporter(open(args.out, "w", encoding="utf-8"))

    print(f"Scanning {args.source} ...")
    with open(args.source, encoding="utf-8", errors="replace") as f:
        while kept < args.games:
            game = chess.pgn.read_game(f)
            if game is None:
                print("  reached end of source file")
                break
            scanned += 1

            if not elo_ok(game, args.min_elo, args.max_elo):
                rejected_elo += 1
                continue

            n_moves = len(list(game.mainline_moves()))
            if n_moves < args.min_moves:
                rejected_short += 1
                continue

            if game_fingerprint(game) in training_ids:
                rejected_overlap += 1
                continue

            game.accept(exporter)
            kept += 1

            if kept % 50 == 0:
                print(f"  ...{kept} games kept ({scanned} scanned)")

    print(f"\nWrote {kept} games to {args.out}")
    print(f"  scanned              : {scanned}")
    print(f"  rejected: rating     : {rejected_elo}")
    print(f"  rejected: too short  : {rejected_short}")
    print(f"  rejected: IN TRAINING: {rejected_overlap}")
    if rejected_overlap:
        print("\n  Overlap was found and excluded -- this is exactly why the")
        print("  fingerprint check exists. Without it those games would have")
        print("  silently inflated your results.")


if __name__ == "__main__":
    main()