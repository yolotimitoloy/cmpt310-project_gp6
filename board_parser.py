import time
import chess
import chess.pgn
from pathlib import Path


def get_training_data_from_folder(folder_path, progress_every=200, max_positions=None):
    path = Path(folder_path)
    pgn_files = [path] if path.is_file() else sorted(path.glob("*.pgn"))
    if not pgn_files:
        print(f"WARNING: no .pgn files found at {folder_path}")

    training_data = []
    start = time.time()
    games_so_far = 0

    for file_path in pgn_files:
        print(f"Parsing {file_path.name} ...")
        with open(file_path, encoding="utf-8", errors="replace") as opened_file:
            games_so_far = get_training_data_from_file(
                opened_file, training_data, progress_every, start, games_so_far, max_positions
            )
        if max_positions is not None and len(training_data) >= max_positions:
            print(f"  Reached max_positions={max_positions}, stopping early "
                  f"(did not parse the rest of the file/folder).")
            break

    elapsed = time.time() - start
    print(f"\nFinished parsing: {games_so_far} games, {len(training_data)} positions "
          f"in {elapsed:.1f}s ({len(training_data)/max(elapsed,1e-9):.0f} positions/sec)")
    return training_data


def get_training_data_from_file(file, training_data, progress_every, start_time, games_so_far, max_positions=None):
    while True:
        if max_positions is not None and len(training_data) >= max_positions:
            break

        game = chess.pgn.read_game(file)
        if game is None:
            break

        board = game.board()
        for move in game.mainline_moves():
            training_data.append((board.copy(), move))
            board.push(move)
            if max_positions is not None and len(training_data) >= max_positions:
                break

        games_so_far += 1
        if games_so_far % progress_every == 0:
            elapsed = time.time() - start_time
            rate = games_so_far / elapsed
            print(f"  ...{games_so_far} games parsed, {len(training_data)} positions so far "
                  f"({rate:.0f} games/sec, {elapsed:.1f}s elapsed)")

    return games_so_far