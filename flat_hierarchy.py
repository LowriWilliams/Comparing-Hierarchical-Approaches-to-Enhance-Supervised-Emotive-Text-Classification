from __future__ import division
import csv

# Read file - csv columns: sentence_id, actual_class, predicted_class

def read_file():  
    with open('j48_testing.csv', 'r') as f:
        reader = csv.reader(f)
        file_as_list = list(reader)

    del file_as_list[0]
    
    return file_as_list


# Get unique actual classes

def get_unique_classes(file_as_list):
    classes = set()

    for i in file_as_list:
        classes.add(i[1])

    return classes

    
# True positive

def true_positive(file_as_list, classes):
    temp = []
    
    for i in classes:
        for j in file_as_list:
            if i == j[1] and i == j[2]:
                temp.append(i)
                
    TP = []
    
    for i in classes:
        counter = temp.count(i)
        TP.append([i, counter])

    return TP
            

# False positive

def false_positive(file_as_list, classes):
    temp = []
    
    for i in classes:
        for j in file_as_list:
            if i == j[1] and i != j[2]:
                temp.append(i)
                
    FP = []
    
    for i in classes:
        counter = temp.count(i)
        FP.append([i, counter])

    return FP
            

# False negative

def false_negative(file_as_list, classes):
    temp = []
    
    for i in classes:
        for j in file_as_list:
            if i != j[1] and i == j[2]:
                temp.append(i)

    FN = []
    
    for i in classes:
        counter = temp.count(i)
        FN.append([i, counter])

    return FN


# Precision

def precision(TP, FP):

    temp = []
    
    for i in TP:
        temp.append(i[1])

    TP_sum = sum(temp)
    
    temp = []

    for i in FP:
        temp.append(i[1])

    FP_sum = sum(temp)

    try:
        precision = (TP_sum / (TP_sum + FP_sum))
    except:
        precision = 0.0

    precision = round(precision, 2)

    return precision


# Recall

def recall(TP, FN):
    temp = []
    
    for i in TP:
        temp.append(i[1])

    TP_sum = sum(temp)
    
    temp = []

    for i in FN:
        temp.append(i[1])

    FN_sum = sum(temp)

    try:
        recall = (TP_sum / (TP_sum + FN_sum))
    except:
        recall = 0.0

    recall = round(recall, 2)
    return recall


# F-measure

def fmeasure(precision, recall):
    try:
        f_measure = (2 * ((precision * recall)/(precision + recall)))
    except:
        f_measure = 0.0

    f_measure = round(f_measure, 2)

    return f_measure


# All results

def all_results(precision, recall, f_measure):
    print("Microaveraged Precision: ", precision)
    print("Microaveraged Recall: ", recall)
    print("Microaveraged F-measure: ", f_measure)



##########################################################
    

# Call functions
def main():
    file_as_list = read_file()
    classes = get_unique_classes(file_as_list)
    TP = true_positive(file_as_list, classes)
    FP = false_positive(file_as_list, classes)
    FN = false_negative(file_as_list, classes)
    P = precision(TP, FP)
    R = recall(TP, FN)
    F = fmeasure(P, R)
    all_results(P, R, F)


if __name__ == "__main__":
    main()