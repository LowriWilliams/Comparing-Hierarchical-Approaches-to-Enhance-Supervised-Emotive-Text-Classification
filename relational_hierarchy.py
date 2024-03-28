from __future__ import division
import csv

# Read file - csv columns: sentence_id, actual_class, predicted_class

def read_file():  
    with open('j48_testing.csv', 'r') as f:
        reader = csv.reader(f)
        file_as_list = list(reader)

    del file_as_list[0]

    with open('affect_map.csv', 'r') as f:
        reader = csv.reader(f)
        emotions = list(reader)
        
    return file_as_list, emotions


# Get unique actual classes

def get_unique_classes(file_as_list, emotions):
    emotion_indicies = []

    for i in file_as_list:
        tmp = [i[0]]
        for j in emotions:
            a = j[1][:10]
            if i[1] == j[0]:
                tmp.append(a)
            if i[2] == j[0]:
                tmp.append(a)
        emotion_indicies.append(tmp)
    
    classes = set()

    for i in emotion_indicies:
        classes.add(i[1])
        
    return classes, emotion_indicies

    
# True positive

def true_positive(emotion_indicies, classes):
    temp = []

    for i in classes:
        for j in emotion_indicies:
            if i == j[1] and i == j[2]:
                temp.append(i)
                
    TP = []
    
    for i in classes:
        counter = temp.count(i)
        TP.append([i, counter])
        
    return TP
            

# False positive

def false_positive(emotion_indicies, classes):
    temp = []
    
    for i in classes:
        for j in emotion_indicies:
            if i == j[1] and i != j[2]:
                temp.append(i)
                
    FP = []
    
    for i in classes:
        counter = temp.count(i)
        FP.append([i, counter])

    return FP
            

# False negative

def false_negative(emotion_indicies, classes):
    temp = []
    
    for i in classes:
        for j in emotion_indicies:
            if i != j[1] and i == j[2]:
                temp.append(i)

    FN = []
    
    for i in classes:
        counter = temp.count(i)
        FN.append([i, counter])

    return FN


# Precision

def precision(TP, FP):

    precision = []

    for a, b in zip(TP, FP):
        precision.append(a[1]/(a[1]+b[1]))

    print('PRECISION ', sum(precision)/len(precision))

    # temp = []
    
    # for i in TP:
    #     temp.append(i[1])

    # TP_sum = sum(temp)
    
    # temp = []

    # for i in FP:
    #     temp.append(i[1])

    # FP_sum = sum(temp)

    # try:
    #     precision = (TP_sum / (TP_sum + FP_sum))
    # except:
    #     precision = 0.0

    # precision = round(precision, 2)

    return precision


# Recall

def recall(TP, FN):

    recall = []

    for a, b in zip(TP, FN):
        recall.append(a[1]/(a[1]+b[1]))

    print('RECALL ', sum(recall)/len(recall))

    # temp = []
    
    # for i in TP:
    #     temp.append(i[1])

    # TP_sum = sum(temp)
    
    # temp = []

    # for i in FN:
    #     temp.append(i[1])

    # FN_sum = sum(temp)

    # try:
    #     recall = (TP_sum / (TP_sum + FN_sum))
    # except:
    #     recall = 0.0

    # recall = round(recall, 2)
    return recall


# F-measure

def fmeasure(precision, recall):

    fmeasure = []

    for a, b in zip(precision, recall):
        try:
            fmeasure.append(2*((a*b)/(a+b)))
        except ZeroDivisionError:
            fmeasure.append(0)

    print('FMEASURE', sum(fmeasure)/len(fmeasure))


    # try:
    #     f_measure = (2 * ((precision * recall)/(precision + recall)))
    # except:
    #     f_measure = 0.0

    # f_measure = round(f_measure, 2)

    # return f_measure


# All results

def all_results(precision, recall, f_measure):
    print("Microaveraged Precision: ", precision)
    print("Microaveraged Recall: ", recall)
    print("Microaveraged F-measure: ", f_measure)



##########################################################
    

# Call functions
def main():
    file_as_list = read_file()[0]
    emotions = read_file()[1]
    classes = get_unique_classes(file_as_list, emotions)[0]
    emotion_indicies = get_unique_classes(file_as_list, emotions)[1]
    TP = true_positive(emotion_indicies, classes)
    FP = false_positive(emotion_indicies, classes)
    FN = false_negative(emotion_indicies, classes)
    P = precision(TP, FP)
    R = recall(TP, FN)
    fmeasure(P, R)
    # all_results(P, R, F)


if __name__ == "__main__":
    main()