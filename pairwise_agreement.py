import csv
import itertools
from itertools import groupby, combinations
import math
import pandas as pd
import numpy as np


# Read csv files. Annotations: IP, sentence_id, annotation

def read_files():
    with open('annotations.csv', 'r') as f:
        reader = csv.reader(f)
        annotations = list(reader)

    return annotations


def calculate_pairwise_agreement(annotations):
    results = []

    for k, g in groupby(annotations, key=lambda x: x[1]):
        for x, y in combinations(g, 2):
            tmp = [x[0], y[0], x[1], x[2], y[2]]
            results.append(tmp)
    
    results_data_frame = pd.DataFrame(results, columns=['IP1', 'IP2', 'sentence id', 'IP1 annotation', 'IP2 annotation'])

    results_data_frame['agreement'] = np.where(results_data_frame['IP1 annotation'] == results_data_frame['IP2 annotation'], 1, 0)

    average_pairwise_agreement = round(((results_data_frame['agreement'].mean())*100),2)

    print("The percentage agreement is", average_pairwise_agreement, "%")

    results_data_frame.to_csv('pairwise agreement results.csv', sep=',',  index=False)



##########################################################
    

# Call functions
def main():
    annotations = read_files()
    calculate_pairwise_agreement(annotations)


if __name__ == "__main__":
    main()



    









  

