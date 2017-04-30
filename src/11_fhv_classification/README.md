### Purpose


Run MLP or LSTM on FHV matrices created in the previous step:

- `init.py` for LSTM
- `init_mlp.py` for MLP

For both models, first run the scripts without the `--doTest` flag first to do parameter search and train the model, then run them with the `--doTest` flag to use the trained models to evaluate on the test set 