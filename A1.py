# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:09:03 2021

@author: Weixu Wang
"""

#%%

import numpy as np
import random
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import pdist, squareform
from tabulate import tabulate
import sys

#%% md

## Reading KOS Data Set

#%%

kosFile = open("docword.kos.txt", "r")

docLines = kosFile.read().splitlines()

numOfDocs = int(docLines[0])
numOfWords = int(docLines[1])

print("Number of words: {0}".format(numOfWords))
print("Number of docs: {0}".format(numOfDocs))

docData = docLines[3:]
#convert into integer
docDataParsed = [list(map(int, doc.split())) for doc in docData]

#%% md

### Pre processing input

#%% md

#In order to lookup words for each doc in constant time, I create a map of docId as key and set of wordsIds as value.

#In order to make it easy for indexes documents, I am going to subtract 1 from doc Ids and word Ids so that they start from 0

#%%

# wordsPerDoc will be a map of docId as key will value as list of wordIds
wordsPerDoc = {}

for doc in docDataParsed:
    # Start from 0 to make it easy to index and loop
    docId = doc[0] - 1
    wordId = doc[1] - 1
    if docId in wordsPerDoc:
        wordsPerDoc[docId].add(wordId)
    else:
        wordsPerDoc[docId] = {wordId}

#%% md

## Calculating All Pair Jaccord Similarity (Brute Force)

#%%

# Calculate Jaccard similarity for 2 given document ids
def calculateJaccardSimilarity(docId1 , docId2, wordsPerDoc):
    wordsDoc1 = wordsPerDoc[docId1]
    wordsDoc2 = wordsPerDoc[docId2]
    commonWords = wordsDoc1.intersection(wordsDoc2)
    return len(commonWords) / len(wordsDoc1 | wordsDoc2)

"""
Calculate numOfDoc * numOfDoc matrix with value at i, j
denoting the jacccord similarity between i and j
"""
def calculateJaccaordSimilarityMatrix(wordsPerDoc, numOfDocs):
    jaccardSimilarityMatrix = np.ones([numOfDocs, numOfDocs])
    for docId1 in range(numOfDocs):
        for docId2 in range(docId1, numOfDocs):
            jaccardSimilarityMatrix[docId1][docId2] = calculateJaccardSimilarity(docId1, docId2, wordsPerDoc)
            jaccardSimilarityMatrix[docId2][docId1] = jaccardSimilarityMatrix[docId1][docId2]
    return jaccardSimilarityMatrix

#%% md

### 1 a) Time taken to calculate jaccord similarity - Brute Force

#%%

startTime = time.time()
jaccardSimilarityMatrix = calculateJaccaordSimilarityMatrix(wordsPerDoc,
                                                            numOfDocs)
print("Time taken to calculate all pair jaccord similarity: {0} seconds".format(time.time() - startTime))

#%% md

### Saving jaccord similarity matrix to a file

#%%

np.savetxt('JaccordBruteForce.txt', jaccardSimilarityMatrix)

#%% md

### 1 b) Calculating Average Jaccard similarity of all pairs except identical pairs:

#%%

# Take matrix sum, ignore identical pairs(substracting numOfDocs for 1 in diagonal) 
(np.sum(jaccardSimilarityMatrix) - numOfDocs)/(numOfDocs*numOfDocs - numOfDocs)

#%% md

## Computing Minhash Signature for all documents with num of hash functions = 10

#%%

def generateHashFunction(numOfWords):
    p = 999983
    a = random.randint(0, p-1)
    b = random.randint(0, p-1)
    def hashFunction(x):
        return ((a*x + b)%p)%numOfWords
    return hashFunction

def generateHashFunctions(numOfHashFunctions, numOfWords):
    return [generateHashFunction(numOfWords) for _ in range(numOfHashFunctions)]

def computeSignatureMatrix(hashFunctions, wordsPerDoc):
    numOfHashFunctions = len(hashFunctions)
    numOfDocs = len(wordsPerDoc)
    signatureMatrix = np.zeros([numOfHashFunctions, numOfDocs])
    for docId in range(numOfDocs):
        for hashFunctionIndex in range(numOfHashFunctions):
            hashFunction = hashFunctions[hashFunctionIndex]
            signatureMatrix[hashFunctionIndex][docId] = min([hashFunction(word) for word in wordsPerDoc[docId]])
    return signatureMatrix

#%% md

### 2 a) Time taken to calculate min hash signature with 10 hash functions

#%%

numOfHashFunctions = 10
startTime = time.time()
hashFunctions = generateHashFunctions(numOfHashFunctions, numOfWords)
signatureMatrix = computeSignatureMatrix(hashFunctions, wordsPerDoc)
print("Time taken to calculate min hash signature using 10 hash functions: {0} seconds".format(time.time() - startTime))

#%% md

## Computing all pairs similarity estimators based on MinHash signature

#%% md

##Computing all pair similarity for numOfHash functions,d = 10, 20, 30 ... 100

#%%

def computeSimilarityEstimate(signature_matrix):
    numOfHashFunctions = len(signature_matrix)
    tranform = np.transpose(signature_matrix)
    similarityEstimate = squareform(1 - pdist(tranform, 'matching'))
    np.fill_diagonal(similarityEstimate, 1)
    return similarityEstimate

def calculateMAE(similarityEstimate, jaccardSimilarityMatrix):
    numOfDocs = len(similarityEstimate)
    return np.sum(abs(similarityEstimate.__sub__(jaccardSimilarityMatrix)))/(numOfDocs**2 - numOfDocs)

hashFunctionCount = []
maeValues = []
timeTaken = []

random.seed(1800)
for numOfHashFunctions in range(10, 101, 10):
    hashFunctions = generateHashFunctions(numOfHashFunctions, numOfWords)
    signatureMatrix = computeSignatureMatrix(hashFunctions, wordsPerDoc)
    
    startTime = time.time()
    similarityEstimate  = computeSimilarityEstimate(signatureMatrix)
    timeTaken.append(time.time() - startTime)
    
    mae = calculateMAE(similarityEstimate, jaccardSimilarityMatrix)
    
    hashFunctionCount.append(numOfHashFunctions)
    maeValues.append(mae)

#%% md

### 3 a) Running time of estimating all pairs similarity based on MinHash with different values of d

#%%

print(tabulate(zip(hashFunctionCount, timeTaken), headers=['Number Of Hash Functions', 'Time in seconds']))

#%% md

### 3 b) Plotting MAE vs number of hash functions (d)

#%%

plt.xlabel("Number of hash functions (d)")
plt.ylabel("Mean absoulte error (MAE)")
plt.plot(hashFunctionCount, maeValues)

#%%

print(tabulate(zip(hashFunctionCount, maeValues), headers=['Number Of Hash Functions', 'Mean Absolute error']))

#%% md

### Choosing r and b for LSH such that  false negatives of 60%- similar pairs at most 10%

#%% md

##In order to choose the r and b values, we can calculate the false negative pobability for differen values of b and r

##Best pair will be the one where the false negative threshold is at most 0.1 and the b value is smallest to avoid false positives.

#%%

def evaluateFalseNegativeProbability(similarityThreshold, numOfBands, numOfRowsInBand):
    return (1 - similarityThreshold**numOfRowsInBand)**numOfBands

bVals = np.array([10, 20, 25, 50])
rVals = np.array([10, 5, 4, 2])

fnProbability = np.zeros(4)
threshold = 0.6
for pairIndex in range(len(bVals)):
    fnProbability[pairIndex] = evaluateFalseNegativeProbability(threshold, bVals[pairIndex], rVals[pairIndex])

print(tabulate(zip(bVals, rVals, fnProbability), headers=['Number of Bands(b)', 'Number of rows per band(r)',
                                                          'False negative Probability']))

#%% md

### 4 a) Best b and r setting for false negatives of 60%- similar pairs to be at most 10%

#%%

bOptimal = min(bVals[(fnProbability <= 0.1)])
rOptimal = int(100 / bOptimal)
print("Optimal pair for b and r: b = {0}, r = {1}".format(bOptimal, rOptimal))

#%% md

### 4 b) Space usage affected by these parameters

#%% md

#Space utilisation will be different for each pair as the space will depend on the number of bands, since for each band we will need to maintain a hash table for hashing the r inteers for each document.

#The space will be proportional to b for my implemntation of the universal hash function for integer list since the size of the hash table will be 10 * Number of Documents.

#So the larger the value of b, the higher space is needed for these hash tables.

#Note: Space usage is approximate, does not take into account space used in chaining.

#%%

spaceBasedOnB = [b*10*numOfDocs*sys.getsizeof(int()) for b in bVals]

print(tabulate(zip(bVals, spaceBasedOnB), headers=['Number Of Bands', 'Approx Space usage in bytes']))

#%% md

### Plotting approximate space used vs number of bands

#%%

plt.xlabel("Number of bands (b)")
plt.ylabel("Approx space used (in bytes) ")
plt.plot(bVals, spaceBasedOnB)

#%% md

### LSH candidate pair calculation with d = 100, b = 25, r =4

#%%

def getUniversalHashFunctionForIntegerList(numOfDocs, r):
    p = 999983
    a = random.sample(range(p), r +1)
    def hashIntegerList(integerList):
        result = a[0]
        for i in range(r):
            result += a[i + 1]*integerList[i]
        return ((result)%p)%(10*numOfDocs)
    return hashIntegerList

"""
For each band computes the hash value for each document based on the r integers in the band.
Returns a list of maps where each map has the hash value as the key and a list of docIds
with the correponding hash value. 
"""
def gethashValueToDocumentListMappingPerBand(signatureMatrix, b, r, numOfDocs):
    hashValueToDocumentListMappingPerBand = []
    for bandNumber in range(b):
        hashValueToDocumentList = {}
        
        hashFunction = getUniversalHashFunctionForIntegerList(numOfDocs, r)
        bandData = signatureMatrix[bandNumber*r:(bandNumber + 1)*r, ]
        hashValues = np.apply_along_axis(hashFunction, 0, bandData)
        
        for docId in range(len(hashValues)):
            if hashValues[docId] in hashValueToDocumentList:
                hashValueToDocumentList[hashValues[docId]].append(docId)
            else:
                hashValueToDocumentList[hashValues[docId]] = [docId]
        
        hashValueToDocumentListMappingPerBand.append(hashValueToDocumentList)
    return hashValueToDocumentListMappingPerBand

"""
Creates pairs for given list
Sample Input: [1,2,3]
Sample Output: [(1,2), (1,3), (2,3)]
"""
def getPairs(docList):
    numOfDocs = len(docList)
    pairs = []
    for docId1 in range(numOfDocs):
        for docId2 in range(docId1 + 1, numOfDocs):
            pairs.append((docList[docId1], docList[docId2]))
    return pairs

"""
Hashes all documents using b bands and r rows per band.
Then for all buckets with multiple documents in each band, generates candidate pairs and 
returns the final set of candidate pairs after removing any duplicates 
"""
def getCandidatePairs(signatureMatrix, b, r, numOfDocs):
    hashValueToDocumentListMappingPerBand = gethashValueToDocumentListMappingPerBand(signatureMatrix, b, r, numOfDocs)
    candidateSet = set()
    for hashValueToDocumentList in hashValueToDocumentListMappingPerBand:
        for hashValue in hashValueToDocumentList:
            if(len(hashValueToDocumentList[hashValue]) >1):
                pairs = getPairs(hashValueToDocumentList[hashValue])
                for pair in pairs:
                    candidateSet.add(pair)
    return candidateSet

#%%

numOfHashFunctions = 100
hashFunctions = generateHashFunctions(numOfHashFunctions, numOfWords)
signatureMatrix = computeSignatureMatrix(hashFunctions, wordsPerDoc)

candidatePairs = getCandidatePairs(signatureMatrix, bOptimal, rOptimal, numOfDocs)

#%% md

### 4 c ) Calculating False candidate ratio for 60% similarity, b = 25, r =4

#%%

falseCandidateCount = 0
for candidatePair in candidatePairs:
    if(jaccardSimilarityMatrix[candidatePair[0]][candidatePair[1]] < 0.6):
        falseCandidateCount = falseCandidateCount + 1

print("False candidate ratio: {0}".format(falseCandidateCount/len(candidatePairs)))

#%% md

### 4 d) Calculating  probability that a dissimilar pair with Jaccard ≤ 0.3 is a candidate pair

#%%

def findDissimilarDocuments(threshold, jaccardSimilarityMatrix):
    numOfDocs = len(jaccardSimilarityMatrix)
    dissimlarPairs = set()
    for docId1 in range(numOfDocs):
        for docId2 in range(docId1 + 1, numOfDocs):
            if (jaccardSimilarityMatrix[docId1][docId2] <= threshold):
                dissimlarPairs.add((docId1, docId2))
    return dissimlarPairs

dissimilarCandidateCount = 0

for candidatePair in candidatePairs:
    if(jaccardSimilarityMatrix[candidatePair[0]][candidatePair[1]] <= 0.3):
        dissimilarCandidateCount = falseCandidateCount + 1
        
dissimilarDocuments = findDissimilarDocuments(0.3, jaccardSimilarityMatrix)
print("Probability that a dissimilar pair with jaccard <= 0.3 is a candidate pair: {0}"
      .format(dissimilarCandidateCount/len(dissimilarDocuments)))

#%% md

### Calculating false negative rate for 0.6 similarity threshold

#%%

def findSimilarDocuments(similarity_threshold, jaccardSimilarityMatrix):
    numOfDocs = len(jaccardSimilarityMatrix)
    similar_pairs = set()
    for docId1 in range(numOfDocs):
        for docId2 in range(docId1 + 1, numOfDocs):
            if (jaccardSimilarityMatrix[docId1][docId2] >= similarity_threshold):
                similar_pairs.add((docId1, docId2))
    return similar_pairs

similarDocSet = findSimilarDocuments(0.6, jaccardSimilarityMatrix)

print("False negative rate: {0}".format(1 - len(set(similarDocSet).intersection(candidatePairs))/len(similarDocSet)))
