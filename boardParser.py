import chess
import chess.pgn
from pathlib import Path

def get_training_data_from_folder(folderPath):

    folder = Path(folderPath)
    trainingData = []

    for file in folder.glob("*.pgn"):
        with open(file) as opennedFile:
            get_training_data_from_file(opennedFile, trainingData)

    return trainingData

def get_training_data_from_file(file, trainingData):

    while True:
        game = chess.pgn.read_game(file)

        if game is None:
            break
        
        node = game
        while node.next() != None:
            boardState = node
            nextMove = game.next().move

        node = node.next()
        trainingData.append((boardState, nextMove))

#data = get_training_data_from_folder("pgnFiles")

#print(len(data))