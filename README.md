# Grokking in Machine Learning

## Experiments

You can find all the codes in the **code** folder, and the possible output results in the **result** folder.

#### Grokking in Transformer

We trained a transformer using AdamW with different alphas under \(p=47\) and \(p=97\).

To redo the example experiment, run this:

    python ./code/Transformer/experiment.py

If you want to do other experiments, change the parameters directly in the codes.

#### Grokking in Other Models

We trained a MLP, a ResNet and a LSTM using AdamW with different alphas under \(p=47\) and \(p=97\).

To redo the example experiments, run these codes:

    python ./code/MLP/experiment.py

    python ./code/ResNet/experiment.py

    python ./code/LSTM/experiment.py

If you want to do other experiments, change the parameters directly in the codes.

#### Different Optimizers

We trained a transformer on with nine optimizers. \(p=97\)

To redo the experiments, run this:

    python ./code/experiment_transformer_optimezers.py

#### Grokking with K >= 2

We did two experiments:
- \(p=17,\alpha=0.3,K=2,3,4\), aimed at showing the different  complexity of the problem.
- \(p=31,\alpha=0.3,K=2,3\), showing how different \(K\) influence grokking.

To redo the experiments, run this:

    python ./code/experiment_transformer_K.py

#### Explanation of grokking
To show how commutativity influence grokking, we trained our transformer on an ordered dataset where the input are increasing, and compared the result with the original total dataset. To redo this, run ths:

    python ./code/experiment_transformer_ordered_data.py
