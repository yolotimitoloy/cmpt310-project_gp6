import boardParser
import numpy as np

def get_processed_data(folderName):
    # Each training example contains one board position and the move played from that position
    data = boardParser.get_training_data_from_folder(folderName)

    # Board state represents 12 x 8 x 8 boolean array, 12 piece-type layers and 8 x 8board coordinates

    for board, move in data:

        oneHotEncoding = get_one_hot()
        mvTup = (move.from_square, move.to_square, piece_to_int(board.board().piece_at(move.from_square).symbol()))

        for square, piece in board.board().piece_map().items():

            layer = piece_to_int(piece.symbol())
            xInd = square % 8
            yInd = square // 8
            

            oneHotEncoding[layer][xInd][yInd] = True
        
        # store all processed examples
        processedData.append((oneHotEncoding, mvTup))

    return processedData

def piece_to_int(piece):

    match piece:
        case 'P':
            return 0
        case 'N':
            return 1
        case 'B':
            return 2
        case 'R':
            return 3
        case 'Q':
            return 4
        case 'K':
            return 5
        case 'p':
            return 6
        case 'n':
            return 7
        case 'b':
            return 8
        case 'r':
            return 9
        case 'q':
            return 10
        case 'k':
            return 11
        
def get_one_hot():

    return np.zeros((12, 8, 8), dtype=bool)
        
#proc = get_processed_data("pgnFiles")

#for dat in proc:
#    print(dat)