# ShallowBlue 
CMPT 310, Summer 2026 Group 6

## Requirements
Python 3.10+ and Stockfish installed as a system binary

## Setup
install files mentioned in requirements.txt`

## Running
 
Trained checkpoints and a held-out PGN are included, so evaluation and inference run without retraining.

**Evaluate the model against Stockfish**
```bash
python run_evaluation.py --positions 200 --depth 10
```
 
Reports top-1 and top-3 agreement and centipawn loss for the MLP, weakened
Stockfish,and a random baseline. A Stockfish-vs-itself self-test runs first to confirm the evaluator is sane.

**Ask model for a move suggestion**
 
```bash
python predict_tim.py model.joblib --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1" --top-k 3
python predict_tim.py model.joblib --moves "e4 c5 Nf3"
```

**Play against the model**
 
```bash
python main.py --player-color white
```
Opens a pygame board; the model plays the other side.

**Retrain from scratch** 
```bash
python train_tim.py pgnFiles --max-positions 100000 --checkpoint-out model.joblib
```

**Rebuild the held-out test set** (optional)
 
```bash
python build_testset.py <source.pgn> --training pgnFiles/test.pgn pgnFiles/test2.pgn \
    --games 300 --out verifSet/heldout.pgn
```

## Files
 
| File | Purpose |
| --- | --- |
| `run_evaluation.py` | Runs Evaluation |
| `stockfish_evaluation.py` | Agreement and centipawn-loss metrics |
| `baseline_models.py` | Stockfish,weakened-Stockfish, and random baselines |
| `adapters.py` | Wraps models behind a common interface for evaluation |
| `train_tim.py` | Trains the MLP from a folder of PGN files |
| `preprocessing.py`, `board_parser.py` | PGN to feature vectors and move labels |
| `predict_tim.py` | Single-position move suggestion |
| `main.py` | Pygame GUI to play against the model |
| `build_testset.py` | Builds the training set |
| `model.joblib` | AI model |
| `verifSet/`, `pgnFiles/` | training data |
