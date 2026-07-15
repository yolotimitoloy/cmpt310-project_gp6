import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


def train_chess_knn(trainingData):
    int k = 1; 

    # Split the dataset into features X and labels y
    x = np.array([
        boardState
        for boardState, move in trainingData
    ])

    y = np.array([
        move
        for boardState, move in trainingData
    ])

    # Train KNN by storing the board positions and labels
    knn = KNeighborsClassifier(
        n_neighbors=min(k, len(x)),
        metric="hamming"
    )

    knn.fit(x, y)

    return knn, x, y

def predict_move(newBoardState, x , y, knn):
    newBoardState = np.asarray(newBoardState)

    # Find exact board-position matches
    exactIndexes = np.where(np.all(x == newBoardState, axis=1))[0]

    if len(exactIndexes) > 0:
        exactMoves = y[exactIndexes]

        # Return the most common move for this exact position
        return Counter(exactMoves).most_common(1)[0][0]

    # No exact match, use KNN
    return knn.predict(newBoardState.reshape(1, -1))[0]

