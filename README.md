# Naive-Bayes-Classifier-from-Scratch
Implementation of the Naive Bayes algorithm, with various improvements, from scratch in Python to classify text abstracts.

## Data
The trg.csv and tst.csv are the training and test datasets respectively. These files contain the abstracts of research papers on the topic of proteins found in one of four classes - Archaea (A), Bacteria (B), Eukaryota (E) or Virus (V).

## Getting Started
To classify the abstracts, run main.py. An output file will be generated (called output.csv), which will contain the generated classifications for the abstracts in tst.csv. Note that the test accuracy for this classifier is 92.6%.

## Details on implementation and improvement
To start, the text data was represented by frequencies of each word in each abstract – the attributes were the number of times each word occurs in each abstract. The Multinomial naïve Bayes classifier was used since the attributes were frequencies instead of categories. To train this classifier, the following formula was used:


![first equation](https://latex.codecogs.com/gif.latex?p_%7Bk%2Ci%7D%3D%5Cfrac%7BN_%7Bk%2Ci%7D&plus;a_i%7D%7BN_k&plus;a%7D)

where *p(k,i)* is the probability of the word *i* given class *k*, *N(k,i)* is the number of times word *i* appears in class *k* in the training set, *N(k)* is the total number of words from class *k*, pseudocount *a(i)* is 1 and *a* is the sum of all *a(i)*. The pseudocount acts as a Laplace Smoother to ensure the denominator is never zero, which would have resulted in an error.

The data also needed to be pre-processed. To do this, any data that could have been potentially irrelevant or noisy was removed. This included the punctuation from all abstracts and words shorter than three characters, as well as any ‘words’ with numeric characters. Several stop words, which are commonly used words, were also removed as they are likely unimportant and will probably not improve the classifier accuracy. 

To extend the naïve Bayes implementation, the inverse document frequency was used to train the classifier instead of the term frequency. The inverse document frequency for a document frequency *df(i,j)* = number of times word *i* appears in document *j*, is given by the following:


![second equation](https://latex.codecogs.com/gif.latex?idf_%7Bi%2Cj%7D%3D%20df_%7Bi%2Cj%7D%20log%28%5Cfrac%7B%5Csum_k1%7D%7B%5Csum_kt_%7Bik%7D%7D%29)

where *t(ik)* is 1 if word *i* appears in training instance *k* and zero otherwise.

To calculate and affirm the accuracy of the classifier, 10-fold cross-validation was used. This was done by first shuffling the training dataset and partitioning it into 10 groups – the partitioning was done manually. Next, the partitions were iterated over using each as the test set and combining the rest as the training set. Prior to the inverse document frequency extension, the overall average accuracy of the classifier was found to be 92.04%. After the extension, the overall average accuracy of the classifier increased to 93.54%.