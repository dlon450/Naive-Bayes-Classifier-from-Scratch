import csv
import numpy as np
import string
import math

# open the trg.csv and tst.csv files and convert them to matrices
with open('trg.csv', newline='') as f:
    reader = csv.reader(f)
    dataset = list(reader) # comment out when you want to find training and test datasets, uncomment next line
#   datafull = list(reader)

with open('tst.csv', newline='') as f:
    reader = csv.reader(f)
    dataset2 = list(reader)

# # shuffle the dataset for cross validation    
# import random
# copy = datafull[1:]
# random.Random(1).shuffle(copy)
# datafull[1:] = copy

# # split into test and training sets - this needs to be done 10 times manually
# dataset = datafull[:3600]
# datatest = datafull[3600:]

# retrieve all words from all abstracts
words = []
for i in range(1,len(dataset)): # maybe start from 1
    abstract = dataset[i][2]
    abstract = abstract.translate(str.maketrans('', '', string.punctuation))
    words_abstract = abstract.split(" ")
    words.extend(words_abstract)

# remove unwanted words
words = list(np.unique(words)) # keeps unique words
length_checker = np.vectorize(len)
words = np.array(words)
words1 = words[np.where(length_checker(words) > 2)] # removes elements with a length <= 2
wordschar = words1[np.char.isalpha(words1)] # removes 'words' that contain digits

stop_words = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]
wordschar = [x for x in wordschar if x not in stop_words] # removes any stop words

# create dictionaries for frequencies of each word
l1 = [([0] * (len(dataset)-1)) for i in range(len(wordschar))]
l2 = [0] * len(wordschar)
counterA = dict(zip(wordschar,l2)) # frequencies of each word in class A
counterB = dict(zip(wordschar,l2)) # frequencies of each word in class B
counterE = dict(zip(wordschar,l2)) # frequencies of each word in class E
counterV = dict(zip(wordschar,l2)) # frequencies of each word in class V

# initialise the dictionary for all words and their corresponding frequencies in each abstract
adf = dict(zip(wordschar,l1))

# add the list of target (class) variables to adf
target = [0] * len(dataset)
for i in range(len(dataset)):
    target[i] = dataset[i][1]
adf["target_var"] = target[1:]

# iterate through all abstracts and count the frequencies of each word
for i in range(1, len(dataset)):
    abstract = dataset[i][2]
    words_abstract = abstract.split(" ")
    
    for j in range(0, len(words_abstract)):
        word = words_abstract[j]
        
        # the word may have been cleaned out of wordschar - may not be in adf
        try:
            adf[word][i-1] = words_abstract.count(word)
        except KeyError:
            continue
            
# inverse document frequencies
for key in adf.keys():
    l = adf[key]
    try:
        # map each the frequencies to its new value
        a = math.log((len(dataset)-1)/np.count_nonzero(l))
        l = map(lambda x: x * a, l)
        adf[key] = list(l)
    except (ZeroDivisionError, TypeError) as error:
        continue

count = [0] * 4 # array for total frequencies of each class in the dataset

# store the frequencies of each word depending on the class in counter variables 
for i in range(1, len(dataset)):
    
    # add the frequency of the word to the corresponding key in counterA
    if adf["target_var"][i-1] == "A":
        count[0] += 1
        for key in adf.keys():
            if key != 'target_var':
                counterA[key] += adf[key][i-1]
                
    elif adf["target_var"][i-1] == "B":
        count[1] += 1
        for key in adf.keys():
            if key != 'target_var':
                counterB[key] += adf[key][i-1]
                
    elif adf["target_var"][i-1] == "E":
        count[2] += 1
        for key in adf.keys():
            if key != 'target_var':
                counterE[key] += adf[key][i-1]
                
    else:
        count[3] += 1
        for key in adf.keys():
            if key != 'target_var':
                counterV[key] += adf[key][i-1]

# remove any words with frequencies of 0
for key in list(counterA):
    if counterA[key] == 0.:
        del counterA[key]        
for key in list(counterB):
    if counterB[key] == 0.:
        del counterB[key]
for key in list(counterE):
    if counterE[key] == 0.:
        del counterE[key]
for key in list(counterV):
    if counterV[key] == 0.:
        del counterV[key]
        
words_count = len(wordschar) # number of words

def prob(data):
    '''
    Returns the conditional probability for each key in a dictionary
    Input: 
        'data' is a dictionary with words as its keys and the frequency of those words as its values
    Output:
        'data2' - dictionary with words as its keys and the conditionary probability for each word
    '''
    total = sum(data.values())
    data2 = data.copy()
    for word,b in data2.items():
        outcome = float((b+1))/float((words_count + total))
        data2[word] = (outcome)
    return data2

# retrieve the probabilities for each class
class_A = prob(counterA)
class_B = prob(counterB)
class_E = prob(counterE)
class_V = prob(counterV)

# retrieve the proportional of each class
classesA = count[0]/(len(dataset)-1)
classesB = count[1]/(len(dataset)-1)
classesE = count[2]/(len(dataset)-1)
classesV = count[3]/(len(dataset)-1)

# log each proportional so that it can be correctly computed and represented
prior1 = math.log(classesB)
prior2 = math.log(classesA)
prior3 = math.log(classesE)
prior4 = math.log(classesV)
table1 = []
table2 = []
table3 = []
table4 = []

# this is for tst.csv
test = [0] * (len(dataset2)-1)
for i in range(1, len(dataset2)):
    test[i-1] = dataset2[i][1]

# # this is for test sets derived from trg.csv - doing cross validation
# test = [0] * len(datatest)
# for i in range(len(datatest)):
#     test[i] = datatest[i][2]

constant = 100

# iterate through abstracts in test dataset
for i in range(0, len(test)):
    data = test[i].split(" ")
    result1 = 1
    result2 = 1
    result3 = 1
    result4 = 1
    
    # iterate through all the words in each abstract
    for j in range(0, len(data)):
        
        # add the logged probability to the result or the constant
        if data[j] in class_A.keys():
            result2 += math.log(class_A.get(data[j]))
        else:
            result2 += constant
            
        if data[j] in class_E.keys():
            result3 += math.log(class_E.get(data[j]))
        else:
            result3 += constant
            
        if data[j] in class_V.keys():
            result4 += math.log(class_V.get(data[j]))
        else:
            result4 += constant
            
        if data[j] in class_B.keys():
            result1 += math.log(class_B.get(data[j]))
        else:
            result1 += constant
            
    # add the result to each corresponding table     
    result1 = result1 + prior1
    table1.append(result1)            
    result2 = result2 + prior2
    table2.append(result2)
    result3 = result3 + prior3
    table3.append(result3)
    result4 = result4 + prior4
    table4.append(result4)

# create new dictionary of logged likelihoods
adf1 = dict()
adf1["A"] = table2
adf1["B"] = table1
adf1["E"] = table3
adf1["V"] = table4

# this is for tst.csv
adf1['Max'] = [0]*(len(dataset2)-1)
class_list = ['A','B','E','V']
# take the minimum logged value and assign the corresponding class to adf1['Max']
for i in range(len(dataset2)-1):
    probs = [adf1['A'][i],adf1['B'][i],adf1['E'][i],adf1['V'][i]]
    adf1['Max'][i] = class_list[probs.index(min(probs))]

# # this is for test dataset from trg.csv
# adf1['Max'] = [0]*(len(datatest))
# class_list = ['A','B','E','V']
# for i in range(len(datatest)):
#     probs = [adf1['A'][i],adf1['B'][i],adf1['E'][i],adf1['V'][i]]
#     adf1['Max'][i] = class_list[probs.index(min(probs))]

# # accuracy of test data 
# datatest_class = [0] * len(datafull[3600:])
# for i in range(len(datafull[3600:])):
#     datatest_class[i] = datafull[3600:][i][1]

# counttst = 0
# for i in range(len(datafull[3600:])):
#     if adf1['Max'][i] != datatest_class[i]:
#         counttst += 1
        
# # this prints the accuracy
# print(str(np.round((1 - counttst/len(datatest_class)) * 100,2)) + "%")

n = 1 # change to 1 for tst.csv and the starting index for test datasets derived from trg.csv
with open('dlon450.csv', mode='w', newline='') as csv_file:
    header_names = ['id', 'class']
    writer = csv.DictWriter(csv_file, fieldnames=header_names)
    writer.writeheader()
    for i in range(len(adf1['Max'])):
        writer.writerow({'id': i+n, 'class': adf1['Max'][i]})