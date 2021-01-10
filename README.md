# PoS-tagging-using-HMM

Dataset: Brown PoS tag corpus. (Attached)
Dataset Format:
● Each line represents one sentence.
● Sentences are already tokenized.
● Words in a line have the format word_tag.

The pos-tagger implements Viterbi algorithm with the following assumptions.
- Markov assumption length 1 : Probability of any state sk depends on its previous state only, i.e., P(sk|sk-1)
- Markov assumption length 2 : Probability of any state sk depends on its previous two states only, i.e., P(sk | sk-2,sk-1 )

Performs 3-fold cross validation on the dataset (Brown_Train.txt) and reports the following for each fold and an average score
1) Precision, recall and F1-score.
2) Tag-wise precision, recall and F1-score
3) Confusion matrix (Each element Aij of matrix A denotes the number of times tag is classified as tag j)
4) Statistics of tag set.
