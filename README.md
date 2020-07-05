# Naive-Bayes-Classifier-from-Scratch
Implementation of the Naive Bayes algorithm, with various improvements, from scratch in Python to classify text abstracts.

## Data
The trg.csv and tst.csv are the training and test datasets respectively. These files contain the abstracts of research papers on the topic of proteins found in one of four classes - Archaea (A), Bacteria (B), Eukaryota (E) or Virus (V).

## Getting Started
To classify the abstracts, run main.py. An output file will be generated (called output.csv), which will contain the generated classifications for the abstracts in tst.csv. Note that the test accuracy for this classifier is 92.6%.

## Details on implementation and improvement
To start, the text data was represented by frequencies of each word in each abstract – the attributes were the number of times each word occurs in each abstract. The Multinomial naïve Bayes classifier was used since the attributes were frequencies instead of categories. To train this classifier, the following formula was used:
![first equation](https://latex.codecogs.com/gif.latex?p_%7Bk%2Ci%7D%3D%5Cfrac%7BN_%7Bk%2Ci%7D&plus;a_i%7D%7BN_k&plus;a%7D)
where ![second equation](https://latex.codecogs.com/gif.latex?p_%7Bk%2Ci%7D) is the probability of the word i given class k, ![third equation](https://latex.codecogs.com/gif.latex?N_%7Bk%2Ci%7D) is the number of times word i appears in class k in the training set, ![](https://latex.codecogs.com/gif.latex?N_k) is the total number of words from class k, pseudocount ![](https://latex.codecogs.com/gif.latex?a_i%20%3D%201) and ![](https://latex.codecogs.com/gif.latex?a_i%20%3D%20%5Csum_i%20a_i). The pseudocount acts as a Laplace Smoother to ensure the denominator is never zero, which would have resulted in an error.