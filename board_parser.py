import time
import chess
import chess.pgn
from pathlib import Path

RESULT_ALIASES = {
    "white": {"1-0"},
    "black": {"0-1"},
    "draw": {"1/2-1/2"},
    "any": None,
}


def _game_passes_filters(game, min_elo, results_allowed):
    if results_allowed is not None and game.headers.get("Result") not in results_allowed:
        return False
    if min_elo is not None:
        try:
            white_elo = int(game.headers.get("WhiteElo", 0))
            black_elo = int(game.headers.get("BlackElo", 0))
        except ValueError:
            return False  # missing/non-numeric Elo -- treat as not meeting the bar
        if white_elo < min_elo or black_elo < min_elo:
            return False
    return True


def get_training_data_from_folder(folder_path, progress_every=200, max_positions=None,
                                   min_elo=None, results_allowed=None):
    path = Path(folder_path)
    pgn_files = [path] if path.is_file() else sorted(path.glob("*.pgn"))
    if not pgn_files:
        print(f"WARNING: no .pgn files found at {folder_path}")

    training_data = []
    start = time.time()
    games_seen = 0
    games_kept = 0

    for file_path in pgn_files:
        print(f"Parsing {file_path.name} ...")
        with open(file_path, encoding="utf-8", errors="replace") as opened_file:
            games_seen, games_kept = get_training_data_from_file(
                opened_file, training_data, progress_every, start,
                games_seen, games_kept, max_positions, min_elo, results_allowed,
            )
        if max_positions is not None and len(training_data) >= max_positions:
            print(f"  Reached max_positions={max_positions}, stopping early.")
            break

    elapsed = time.time() - start
    print(f"\nFinished parsing: {games_seen} games seen, {games_kept} kept after filters, "
          f"{len(training_data)} positions in {elapsed:.1f}s "
          f"({len(training_data)/max(elapsed,1e-9):.0f} positions/sec)")
    return training_data


def get_training_data_from_file(file, training_data, progress_every, start_time,
                                 games_seen, games_kept, max_positions=None,
                                 min_elo=None, results_allowed=None):
    while True:
        if max_positions is not None and len(training_data) >= max_positions:
            break

        game = chess.pgn.read_game(file)
        if game is None:
            break
        games_seen += 1

        if _game_passes_filters(game, min_elo, results_allowed):
            games_kept += 1
            board = game.board()
            for move in game.mainline_moves():
                training_data.append((board.copy(), move))
                board.push(move)
                if max_positions is not None and len(training_data) >= max_positions:
                    break

        if games_seen % progress_every == 0:
            elapsed = time.time() - start_time
            rate = games_seen / elapsed
            print(f"  ...{games_seen} games seen ({games_kept} kept), "
                  f"{len(training_data)} positions so far "
                  f"({rate:.0f} games/sec, {elapsed:.1f}s elapsed)")

    return games_seen, games_kept