import argparse
import sys

import pygame
import chess
import joblib

from predict_tim import suggest_moves

# Constants
WIDTH, HEIGHT = 640, 640
SQ_SIZE = WIDTH // 8

WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
HIGHLIGHT = (100, 255, 100)
AI_THINKING_COLOR = (255, 255, 255)

PIECE_UNICODE = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--player-color",
        choices=["white", "black"],
        default="white",
        help="Which side the human plays. The AI automatically loads the "
             "model for the opposite side."
    )
    parser.add_argument(
        "--white-checkpoint",
        default="chess_model_white_2000.joblib",
        help="Path to the model trained to play White (used when the human plays Black)"
    )
    parser.add_argument(
        "--black-checkpoint",
        default="chess_model_black_2000.joblib",
        help="Path to the model trained to play Black (used when the human plays White)"
    )
    return parser.parse_args()


def draw_board(screen, selected_square=None):
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BROWN
            pygame.draw.rect(
                screen,
                color,
                (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            )

    if selected_square is not None:
        rank = 7 - chess.square_rank(selected_square)
        file = chess.square_file(selected_square)

        pygame.draw.rect(
            screen,
            HIGHLIGHT,
            (file * SQ_SIZE, rank * SQ_SIZE, SQ_SIZE, SQ_SIZE),
            5
        )


def draw_pieces(screen, board, font):
    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece:
            rank = 7 - chess.square_rank(square)
            file = chess.square_file(square)

            text = font.render(
                PIECE_UNICODE[piece.symbol()],
                True,
                (0, 0, 0)
            )

            rect = text.get_rect(
                center=(
                    file * SQ_SIZE + SQ_SIZE // 2,
                    rank * SQ_SIZE + SQ_SIZE // 2
                )
            )

            screen.blit(text, rect)


def draw_status(screen, font_small, message):
    if not message:
        return
    text = font_small.render(message, True, (255, 0, 0))
    rect = text.get_rect(center=(WIDTH // 2, 20))
    # Small backing box so the text is legible over the board
    bg = pygame.Rect(rect.left - 6, rect.top - 2, rect.width + 12, rect.height + 4)
    pygame.draw.rect(screen, (255, 255, 255), bg)
    screen.blit(text, rect)


def mouse_to_square(pos):
    x, y = pos
    file = x // SQ_SIZE
    rank = 7 - (y // SQ_SIZE)
    return chess.square(file, rank)


def check_game_over(board):
    if board.is_checkmate():
        winner = "Black" if board.turn else "White"
        return f"Checkmate! {winner} wins."
    if board.is_stalemate():
        return "Stalemate!"
    if board.is_insufficient_material():
        return "Draw (insufficient material)."
    if board.can_claim_fifty_moves():
        return "Draw (50-move rule)."
    if board.can_claim_threefold_repetition():
        return "Draw (threefold repetition)."
    return None


def make_ai_move(clf, board):
    """Ask the model for its best legal move and play it. Returns a status string or None."""
    scored = suggest_moves(clf, board, top_k=1)

    if not scored:
        return "AI has no legal moves."

    move, prob = scored[0]
    san = board.san(move)
    board.push(move)
    print(f"AI plays {san} ({prob:.1%})")
    return None


def main():
    args = parse_args()

    human_color = chess.WHITE if args.player_color == "white" else chess.BLACK

    # The AI plays whichever side the human didn't pick, so load the
    # matching checkpoint for that side.
    ai_checkpoint = args.black_checkpoint if human_color == chess.WHITE else args.white_checkpoint
    ai_side = "Black" if human_color == chess.WHITE else "White"

    print(f"Human plays {args.player_color.capitalize()}. "
          f"Loading {ai_side} AI model from {ai_checkpoint} ...")
    try:
        clf = joblib.load(ai_checkpoint)
    except Exception as e:
        print(f"Failed to load model checkpoint '{ai_checkpoint}': {e}")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Python Chess vs AI")

    font = pygame.font.SysFont("segoe ui symbol", 48)
    font_small = pygame.font.SysFont("arial", 22, bold=True)

    board = chess.Board()

    selected_square = None
    running = True
    status_message = None

    def redraw(thinking=False):
        draw_board(screen, selected_square)
        draw_pieces(screen, board, font)
        if thinking:
            draw_status(screen, font_small, "AI is thinking...")
        elif status_message:
            draw_status(screen, font_small, status_message)
        pygame.display.flip()

    # If the AI plays White, let it move first.
    if board.turn != human_color and not check_game_over(board):
        redraw(thinking=True)
        status_message = make_ai_move(clf, board)
        status_message = status_message or check_game_over(board)

    redraw()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Ignore clicks once the game is over or it's not the human's turn
                if check_game_over(board) or board.turn != human_color:
                    continue

                square = mouse_to_square(event.pos)

                if selected_square is None:
                    piece = board.piece_at(square)

                    if piece and piece.color == board.turn:
                        selected_square = square

                else:
                    move = chess.Move(selected_square, square)

                    # Auto-promote pawns to queen
                    piece = board.piece_at(selected_square)

                    if (
                        piece
                        and piece.piece_type == chess.PAWN
                        and chess.square_rank(square) in [0, 7]
                    ):
                        move = chess.Move(
                            selected_square,
                            square,
                            promotion=chess.QUEEN
                        )

                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None

                        status_message = check_game_over(board)

                        # Let the model respond, if the game isn't over
                        if not status_message and board.turn != human_color:
                            redraw(thinking=True)
                            status_message = make_ai_move(clf, board)
                            status_message = status_message or check_game_over(board)
                    else:
                        selected_square = None

        redraw()

    pygame.quit()


if __name__ == "__main__":
    main()