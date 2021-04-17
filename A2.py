# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:28:41 2021

@author: Weixu Wang
"""

import random

DOC_COUNT = 3430 #hardcoded since we already knew the values for this assignment beforehand 
WORD_COUNT = 6906

K = 400 #K size for MISRA GRIES
P = 19540663 #using https://primes.utm.edu/nthprime/index.php#nth
D = 8 #number of hash functions for COUNTMINSKETCH
W = 7200 #size of counters for COUNTMINSKETCH

AB = [(random.randrange(0, P), random.randrange(0, P)) for i in range(D)] # a and b values for the hash functions

WORD_STREAM = [] #list simulating the stream of words coming in from kos.txt (I preferred to work with the list than with the input stream itself even though this would be impractical in an actual scenario)
RESERVOIR_SAMPLE = [1 for x in range(10000)] #10000 element array for sampling
MISRA_GRIES = {} # MISRA_GRIES[docID] = freq
COUNTER = [[0 for j in range(W)] for i in range(D)] #COUNTER[i][w] to access slot w for hash function i


def process():
    with open('kos.txt') as myfile: #All really straightforward stuff - reads line by line and generates a list of lists. I deleted the first 3 lines of my .txt file. 
        line = myfile.readline()
        while line:
            tokens = line.split()
            word = tokens[1]
            WORD_STREAM.append(word) #In a practical scenario we couldn't do this but this allows me to use a loop rather than a file input stream for the remainder of this code. 
            line = myfile.readline()

def count_frequency(word_list): #A function to return a list of (word, freq) pairs, sorted by frequency (descending)
    freq_dict = {}
    for word in word_list:
        if word in freq_dict:
            freq_dict[word] += 1
        else:
            freq_dict[word] = 1
    return sorted(freq_dict.items(), key = lambda item : item[1], reverse = True)

def bruteforce():
    count = 0
    frequency_sum = 0
    #print('\nStart Bruteforce\n')
    with open('bruteforce.txt', 'w') as write_object:
        for k,v in count_frequency(WORD_STREAM):
            count += 1
            frequency_sum += v
            write_object.write('{} {}\n'.format(k, v))
    #print('\nConcluded Bruteforce with an average frequency of {:.4f}\n'.format(frequency_sum/count))

def resevoir_sampling():
    global RESERVOIR_SAMPLE
    updates = 0
    RESERVOIR_SAMPLE = WORD_STREAM[:10000] #yoink the first 10k elements on the stream.
    for i in range(10000, len(WORD_STREAM)):
        prob = 10000 / (i + 1) #equivalent to s/m
        if random.random() < prob: #basically "with probability s/m"
            updates +=1
            RESERVOIR_SAMPLE[random.randrange(0, 10000)] = WORD_STREAM[i] #set a random element of the sample to the current element of the stream
    
    print('\nStart Resevoir\n')
    for k,v in count_frequency(RESERVOIR_SAMPLE):
        print(k, v)
    print('\nConcluded Reservoir having used {} updates.\n'.format(updates))


def misra_gries():
    count = 0 
    decrements = 0
    for word in WORD_STREAM:
        if word in MISRA_GRIES: #Case A - we increase an existing counter
            MISRA_GRIES[word] += 1
        elif count < K: #Case B - we create a new counter
            MISRA_GRIES[word] = 1
            count += 1
        else: #Case C AKA a decrement step - we decrease all entries by one and remove zeroed out counters. 
            decrements += 1
            for k,v in list(MISRA_GRIES.items()): #list() convert to avoid annoying python types
                v -= 1
                if v == 0:
                    MISRA_GRIES.pop(k)
                    count -= 1
                else:
                    MISRA_GRIES[k] = v
    
    print('\nStart Misra-Gries\n')
    element_count = 0
    for k,v in sorted(MISRA_GRIES.items(), key = lambda item : item[1], reverse = True):
        element_count += v
        print(k, v)
    print('\nConluded Misra-Gries using {} decrement steps and {} elements in the summary\n'.format(decrements, element_count))
        
def run_count_min_sketch():
    for word in WORD_STREAM: 
        for i in range(D): #for every function work out which slot in its corresponding counter array we want to increment
            a,b = AB[i]
            h = ((a * int(word) + b) % P) % W
            COUNTER[i][h] += 1

def query_count_min_sketch(word): #we get a list of all slots for this woord across the hash functions and take their min value
    temp = []
    for i in range(D): 
        a,b = AB[i]
        h = ((a * int(word) + b) % P) % W
        temp.append(COUNTER[i][h])
    return min(temp)


        
        

    



process()
bruteforce()
resevoir_sampling()
misra_gries()
run_count_min_sketch()
