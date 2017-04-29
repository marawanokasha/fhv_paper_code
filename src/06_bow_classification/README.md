### Purpose

Run SVM or MLP on BOW matrices created in previous steps:

- `init.py` for SVM
- `init_mlp.py` for MLP

For MLP, first run the `init_mlp.py` without the `--doTest` flag first to do parameter search and train the model, then run it with the `--doTest` flag to use the trained models to evaluate on the test set 