import csv
import itertools
from itertools import groupby, combinations
import math
import pandas as pd
import numpy as np


# Read csv files. Annotations: IP, sentence_id, annotation

def read_files():
    with open('affect_map.csv', 'r') as f:
        reader = csv.reader(f)
        emotions = list(reader)
    
    with open('annotations.csv', 'r') as f:
        reader = csv.reader(f)
        annotations = list(reader)

    return emotions, annotations


# Map annotations to hierarcihcal index

def map_annotations_to_index(emotions, annotations):
    for i in emotions:
        for j in annotations:
            if j[2] in i:
                j.insert(3, i[1])

    return annotations


# Generalise annotation hierarchical indicies

def generalise_emotions(emotions, mapped_annotations):
    unique_indicies = []

    for i in emotions:
        if i[1] not in unique_indicies:
            unique_indicies.append(i[1])
    
    generalised_emotions = []

    for i in mapped_annotations:
        tmp = [i[0], i[1]]
        for j in unique_indicies:
            if i[3].startswith(j):
                tmp.append(j)
        generalised_emotions.append(tmp)
    
    return generalised_emotions, unique_indicies


# Create feature vector

def create_feature_vectors(generalised_emotions, unique_indicies):
    seperated_indicies = []

    for i in generalised_emotions:
        for j in i[2:]:
            tmp = [i[0], i[1], j]
            seperated_indicies.append(tmp)
    
    feature_vectors = []

    for k, g in groupby(seperated_indicies, lambda x: x[:2]):
        values = [x[2] for x in g]
        tmp = (k + [", ".join("1" if x in values else "0" for x in unique_indicies)])
        feature_vectors.append(tmp)

    return feature_vectors


# Calculate cosine similarity 

def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y

    return sumxy/math.sqrt(sumxx * sumyy)


def calculate_cosine_similarity(feature_vectors):
    f_vector_list = []

    for ip, sentence_id, vector in feature_vectors:
        feature_vector_int = []
        for s in vector.split(','):
            feature_vector_int.append(int(s))
        tmp = [ip, sentence_id, feature_vector_int]
        f_vector_list.append(tmp)

    results = []

    for k, g in groupby(f_vector_list, key=lambda x: x[1]):
        for x, y in combinations(g, 2):
            tmp = [x[0], y[0], x[1], cosine_similarity(x[2], y[2])]
            results.append(tmp)
    
    results_data_frame = pd.DataFrame(results, columns=['IP1', 'IP2', 'sentence id', 'cosine similarity'])

    return results_data_frame


# Include annotation in results and save to csv

def results(results_data_frame, annotations):
    for i in annotations:
        del i[3:]
    
    annotations_data_frame = pd.DataFrame(annotations, columns=["IP", "sentence id", "annotation"])

    results_data_frame = pd.merge(results_data_frame, annotations_data_frame, left_on=['IP1', 'sentence id'], right_on=['IP', 'sentence id'], how='inner')  
    results_data_frame = results_data_frame.rename(columns={'annotation': 'IP1 annotation'})

    results_data_frame = pd.merge(results_data_frame, annotations_data_frame, left_on=['IP2', 'sentence id'], right_on=['IP', 'sentence id'], how='inner')  
    results_data_frame = results_data_frame.rename(columns={'annotation': 'IP2 annotation'})

    results_data_frame = results_data_frame[['IP1', 'IP2', 'IP1 annotation', 'IP2 annotation', 'sentence id', 'cosine similarity']]

    results_data_frame['results >= 0.5'] = np.where(results_data_frame['cosine similarity']>= 0.5, 1, 0)

    average_cosine = round(((results_data_frame['results >= 0.5'].mean())*100),2)

    print("The percentage agreement is", average_cosine, "%")

    results_data_frame.to_csv('cosine similarity results.csv', sep=',',  index=False)



##########################################################
    

# Call functions
def main():
    emotions = read_files()[0]
    annotations = read_files()[1]
    mapped_annotations = map_annotations_to_index(emotions, annotations)
    generalise = generalise_emotions(emotions, mapped_annotations)[0]
    indicies = generalise_emotions(emotions, mapped_annotations)[1]
    feature_vectors = create_feature_vectors(generalise, indicies)
    cosine = calculate_cosine_similarity(feature_vectors)
    results(cosine, annotations)


if __name__ == "__main__":
    main()



    









  

