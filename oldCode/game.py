import pygame
import chess

# Constants
WIDTH, HEIGHT = 640, 640
SQ_SIZE = WIDTH // 8

WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
HIGHLIGHT = (100, 255, 100)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Python Chess")

font = pygame.font.SysFont("segoe ui symbol", 48)
board = chess.Board()

PIECE_UNICODE = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}


def draw_board(selected_square=None):
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


def draw_pieces():
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


def mouse_to_square(pos):
    x, y = pos
    file = x // SQ_SIZE
    rank = 7 - (y // SQ_SIZE)
    return chess.square(file, rank)


selected_square = None
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
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

                    if board.is_checkmate():
                        print("Checkmate!")
                    elif board.is_stalemate():
                        print("Stalemate!")

                selected_square = None

    draw_board(selected_square)
    draw_pieces()

    pygame.display.flip()

pygame.quit()