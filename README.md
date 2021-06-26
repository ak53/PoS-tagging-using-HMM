# PoS-tagging-using-HMM

Dataset: Brown PoS tag corpus. 
Dataset Format:
1) Each line represents one sentence.
2) Sentences are already tokenized.
3) Words in a line have the format <word>_<tag>.

The pos-tagger implements Viterbi algorithm with the following assumptions.
- Markov assumption length 1 : Probability of any state S(k) depends on its previous state only, i.e., P(S(k)|S(k-1))
- Markov assumption length 2 : Probability of any state S(k) depends on its previous two states only, i.e., P(S(k) | S(k-2),S(k-1))

Performs 3-fold cross validation on the dataset and reports the following for each fold and an average score
1) Precision, recall and F1-score.
2) Tag-wise precision, recall and F1-score
3) Confusion matrix (Each element Aij of matrix A denotes the number of times tag is classified as tag j)
4) Statistics of tag set.
