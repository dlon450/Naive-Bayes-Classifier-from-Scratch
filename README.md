# Naive-Bayes-Classifier-from-Scratch
Implementation of the Naive Bayes algorithm, with various improvements, from scratch in Python to classify text abstracts.

## Data
The trg.csv and tst.csv are the training and test datasets respectively. These files contain the abstracts of research papers on the topic of proteins found in one of four classes - Archaea (A), Bacteria (B), Eukaryota (E) or Virus (V).

## Getting Started
To classify the abstracts, run main.py. An output file will be generated (called output.csv), which will contain the generated classifications for the abstracts in tst.csv. Note that the test accuracy for this classifier is 92.6%.

## Details on implementation and improvement
\begin{equation*}
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
\end{equation*}

To start, the text data was represented by frequencies of each word in each abstract – the attributes were the number of times each word occurs in each abstract. The Multinomial naïve Bayes classifier was used since the attributes were frequencies instead of categories. To train this classifier, the following formula was used: 
p_(k,i)=(N_(k,i)+a_i)/(N_k+a)
where p_(k,i) is the probability of the word i given class k, N_(k,i) is the number of times word i appears in class k in the training set, N_k is the total number of words from class k, pseudocount a_i = 1 and a_i = ∑_i▒a_i . The pseudocount acts as a Laplace Smoother to ensure the denominator is never zero, which would have resulted in an error.

The data also needed to be pre-processed. To do this, any data that could have been potentially irrelevant or noisy was removed. This included the punctuation from all abstracts and words shorter than three characters, as well as any ‘words’ with numeric characters. Several stop words, which are commonly used words, were also removed as they are likely unimportant and will probably not improve the classifier accuracy. 
To extend the naïve Bayes implementation, the inverse document frequency was used to train the classifier instead of the term frequency. The inverse document frequency for a document frequency 〖df〗_(i,j) = number of times word i appears in document j, is given by the following:
〖idf〗_(i,j)= 〖df〗_(i,j) log (∑_k▒1)/(∑_k▒t_ik )
where  t_ik is 1 if word i appears in training instance k and zero otherwise.

To calculate and affirm the accuracy of the classifier, 10-fold cross-validation was used. This was done by first shuffling the training dataset and partitioning it into 10 groups – the partitioning was done manually. Next, the partitions were iterated over using each as the test set and combining the rest as the training set. Prior to the inverse document frequency extension, the overall average accuracy of the classifier was found to be 92.04%. After the extension, the overall average accuracy of the classifier increased to 93.54%.