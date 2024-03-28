import csv
import itertools
from itertools import groupby, combinations
import collections
import pandas as pd
import math


# Read file - csv columns: sentence_id, actual_class, predicted_class

def read_file():  
    with open('smo_validation.csv', 'r') as f:
        reader = csv.reader(f)
        file_as_list = list(reader)

    del file_as_list[0]

    with open('affect_map.csv', 'r') as f:
        reader = csv.reader(f)
        emotions = list(reader)
        
    return file_as_list, emotions


# Map emotion to index

def map_emotion_to_index(file_as_list, emotions):
    emotion_to_indicies = []
    actual_classes = []

    for i in file_as_list:
        if i[1] not in actual_classes:
            actual_classes.append(i[1])
        if i[2] not in actual_classes:
            actual_classes.append(i[2])
     
        tmp = [i[0]]
        for j in emotions:
            if i[1] == j[0]:
                tmp.append(j)
            if i[2] == j[0]:
                tmp.append(j)    
        emotion_to_indicies.append(tmp)        

    actual_class_indices = []

    for i in actual_classes:
        tmp = [i]
        for j in emotions:
            if i == j[0]:
              tmp.append(j[1])
        actual_class_indices.append(tmp)

    return emotion_to_indicies, actual_class_indices          


# Generalise annotation hierarchical indicies

def generalise_emotions(emotions, emotions_to_indices, actual_class_indices):    
    unique_indicies = []

    for i in emotions:
        if i[1] not in unique_indicies:
            unique_indicies.append(i[1])
    
    actual_generalised_indicies = []

    for i in actual_class_indices:
        tmp = [i[0]]
        for j in unique_indicies:
            if i[1].startswith(j):
                tmp.append(j)
        actual_generalised_indicies.append(tmp)

    generalised_emotions = []

    for i in emotions_to_indices:
        tmp1 = [i[0]]
        tmp2 = [i[1][0]]
        tmp3 = [i[2][0]]
        for j in unique_indicies:
            if i[1][1].startswith(j):
                tmp2.append(j)
            if i[2][1].startswith(j):
                tmp3.append(j)
        tmp1.append(tmp2)
        tmp1.append(tmp3)                

        generalised_emotions.append(tmp1)
    
    return generalised_emotions, actual_generalised_indicies, unique_indicies



# Create feature vector

def create_feature_vectors(generalised_emotions, actual_generalised_indicies, unique_indicies):
    actual_class_seperated_indicies = []

    for i in actual_generalised_indicies:
        for j in i[1:]:
            tmp = [i[0], j]
            actual_class_seperated_indicies.append(tmp)

    actual_class_feature_vectors = []

    for k, g in groupby(actual_class_seperated_indicies, lambda x: x[:1]):
        values = [x[1] for x in g]
        tmp = (k + [", ".join("1" if x in values else "0" for x in unique_indicies)])
        actual_class_feature_vectors.append(tmp)
    
    actual_classes_2 = []
    predicted_classes = []

    for i in generalised_emotions:
        tmp1 = [i[0]]
        tmp2 = [i[0]]
        for j in i[1]:
            tmp1.append(j)
        for j in i[2]:
            tmp2.append(j)
        actual_classes_2.append(tmp1)
        predicted_classes.append(tmp2)
    
    actual_class_indicies = []

    for i in actual_classes_2:
        for j in i[2:]:
            tmp = [i[0], i[1], j]
            actual_class_indicies.append(tmp)

    actual_feature_vectors = []

    for k, g in groupby(actual_class_indicies, lambda x: x[:2]):
        values = [x[2] for x in g]
        tmp = (k + [", ".join("1" if x in values else "0" for x in unique_indicies)])
        actual_feature_vectors.append(tmp)
    
    predicted_class_indicies = []

    for i in predicted_classes:
        for j in i[2:]:
            tmp = [i[0], i[1], j]
            predicted_class_indicies.append(tmp)

    predicted_feature_vectors = []

    for k, g in groupby(predicted_class_indicies, lambda x: x[:2]):
        values = [x[2] for x in g]
        tmp = (k + [", ".join("1" if x in values else "0" for x in unique_indicies)])
        predicted_feature_vectors.append(tmp)

    return actual_class_feature_vectors, actual_feature_vectors, predicted_feature_vectors
 
    
# Calculate cosine similarity 

def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y

    return sumxy/math.sqrt(sumxx * sumyy)


def calculate_cosine_similarity_actual_predicted(actual_feature_vectors, predicted_feature_vectors):    
    for i in actual_feature_vectors:
        tmp = []
        for s in i[2].split(','):
            tmp.append(int(s))
        del i[2]
        i.append(tmp)
    
    for i in predicted_feature_vectors:
        tmp = []
        for s in i[2].split(','):
            tmp.append(int(s))
        del i[2]
        i.append(tmp)

    actual_predicted_cosine_results = []
    
    for a, b in zip(actual_feature_vectors, predicted_feature_vectors):
        tmp = [a[0], a[1], b[1], cosine_similarity(a[2], b[2])]
        actual_predicted_cosine_results.append(tmp)
    
    return actual_predicted_cosine_results


def calculate_cosine_similarity_actual(actual_class_feature_vectors):
    for i in actual_class_feature_vectors:
        tmp = []
        for s in i[1].split(','):
            tmp.append(int(s))
        del i[1]
        i.append(tmp)
    
    actual_class_cosine_results = []
    
    for i in combinations(actual_class_feature_vectors, 2):
        tmp = [i[0][0], i[1][0], cosine_similarity(i[0][1], i[1][1])]
        actual_class_cosine_results.append(tmp)
    
    return actual_class_cosine_results


# Calculate ACS

def calculate_ACS(actual_classes, actual_predicted_cosine_results):
    M = len(actual_classes)

    all_cosine = 0

    for i in actual_predicted_cosine_results:
        all_cosine += i[2]
    
    ACS = ((2 * all_cosine)/(M * (M - 1)))

    return ACS


# Calculate Con_dj_ci and r_con

def calculate_con_x_cp(ACS, actual_predicted_cosine_results):    
    for i in actual_predicted_cosine_results:
        con_x_cp = ((i[3] - ACS) / (1 - ACS))
        rcon_x_cp = (min(1, (max(-1, con_x_cp))))
        i.append(rcon_x_cp)
       
    return actual_predicted_cosine_results


 # Calculate FP contribution

def calculate_FP_contribution(actual_predicted_cosine_results, actual_classes):
    FP_i_list = []
    FP_con_list = []

    for i in actual_classes:
        FP_i = [i[0]]
        tmp = []
        FP_con_count = 0
        for c in actual_predicted_cosine_results:
            if i[0] == c[1] and i[0] != c[2]:
                if c[2] not in tmp:
                    FP_con_count += c[4]     
                    tmp.append(c[2])
        FP_i.append(tmp)

        FP_con = [i[0], FP_con_count]
        FP_con_list.append(FP_con)
        FP_i_list.append(FP_i)
    
    for i in FP_i_list:
        try:
            count = len(i[1])
            i.append(count)
            del i[1]
        except:
            i.append(0)

    return FP_i_list, FP_con_list


# Calculate FN contribution

def calculate_FN_contribution(actual_predicted_cosine_results, actual_classes):    
    FN_i_list = []
    FN_con_list = []

    for i in actual_classes:
        FN_i = [i[0]]
        tmp = []
        FN_con_count = 0
        for c in actual_predicted_cosine_results:
            if i[0] == c[2] and i[0] != c[1]:
                if c[1] not in tmp:
                    FN_con_count += c[4]
                    tmp.append(c[1])
        FN_i.append(tmp)
                    
        FN_con = [i[0], FN_con_count]
        FN_con_list.append(FN_con)
        FN_i_list.append(FN_i)
    
    for i in FN_i_list:
        try:
            count = len(i[1])
            i.append(count)
            del i[1]
        except:
            i.append(0)
        
    return FN_i_list, FN_con_list


# Calculate TP

def calculate_TP(actual_classes, actual_predicted_cosine_results):
    TP_i_list = []
    
    for i in actual_classes:
        tmp = []
        TP_i = [i[0]]
        for c in actual_predicted_cosine_results:
            if i[0] == c[1] and i[0] == c[2]:
                if c[1] not in tmp:
                    tmp.append(c[1])
        TP_i.append(tmp)
        TP_i_list.append(TP_i)
    
    for i in TP_i_list:
        try:
            count = len(i[1])
            i.append(count)
            del i[1]
        except:
            i.append(0)

    return TP_i_list
    

# Calculate P, R, F

def calculate_P_R_F(actual_classes, TP_i_list, FP_i_list, FP_con_list, FN_i_list, FN_con_list):
    results = []
    p1_tmp = []
    p2_tmp = []
    r1_tmp = []
    r2_tmp = []

    for i in actual_classes:
        for a, b, c, d, e in zip(TP_i_list, FP_i_list, FP_con_list, FN_i_list, FN_con_list):
            if i[0] == a[0] and i[0] == b[0] and i[0] == c[0] and i[0] == d[0] and i[0] == e[0]:
                p_top = float(max(0, (a[1] + c[1] + e[1])))
                p_bottom = float(a[1] + b[1] + e[1])
                try:
                    p = round(((p_top/p_bottom)*100), 2)
                except:
                    p = 0.0
                p1_tmp.append(p_top)
                p2_tmp.append(p_bottom)

                r_top = float(max(0, (a[1] + c[1] + e[1])))
                r_bottom = float(a[1] + d[1] + c[1])
                try:
                    r = round(((r_top/r_bottom)*100), 2)
                except:
                    r = 0.0
                r1_tmp.append(r_top)
                r2_tmp.append(r_bottom)

                try:
                    f = round((2 * ((p * r)/(p + r))), 2)
                except:
                    f = 0.0
 
                tmp = [i[0], a[1], b[1], c[1], d[1], e[1], p, r, f]
        results.append(tmp)

    p_micro = round(((sum(p1_tmp))/(sum(p2_tmp))*100), 2)
    r_micro = round(((sum(r1_tmp))/(sum(r2_tmp))*100), 2)
    f_micro = round((2 * ((p_micro * r_micro)/(p_micro + r_micro))), 2)

    overall_results = [p_micro, r_micro, f_micro]
    
    return results, overall_results


# Create and save dataframe to csv

def create_dataframe(results, overall_results):
    titles = ['class', "TP_i", "FP_i", "FP_con", "FN_i", "FN_con", "P", "R", "F"]

    data_frame = pd.DataFrame.from_records(results, columns=titles)

    print("\n")
    print(data_frame)

    print("\n")
    print("Microaveraged Precision: ", overall_results[0])
    print("Microaveraged Recall: ", overall_results[1])
    print("Microaveraged F-measure: ", overall_results[2])

    data_frame.to_csv("similarity_based_classification_results_smo_validation.csv", sep=',', index=False)
    
    print("\nResults have been saved as a csv file!\n")



##########################################################
    

# Call functions
def main():
    file_as_list = read_file()[0]
    emotions = read_file()[1]
    emotion_to_indicies = map_emotion_to_index(file_as_list, emotions)[0]
    actual_class = map_emotion_to_index(file_as_list, emotions)[1]
    generalise = generalise_emotions(emotions, emotion_to_indicies, actual_class)[0]
    actual_generalised = generalise_emotions(emotions, emotion_to_indicies, actual_class)[1]
    indicies = generalise_emotions(emotions, emotion_to_indicies, actual_class)[2]
    actual_class_vectors = create_feature_vectors(generalise, actual_generalised, indicies)[0]
    actual_vectors = create_feature_vectors(generalise, actual_generalised, indicies)[1]
    predicted_vectors = create_feature_vectors(generalise, actual_generalised, indicies)[2]
    all_cosine = calculate_cosine_similarity_actual_predicted(actual_vectors, predicted_vectors)
    actual_cosine = calculate_cosine_similarity_actual(actual_class_vectors)
    ACS = calculate_ACS(actual_class, actual_cosine)
    con_x_cp = calculate_con_x_cp(ACS, all_cosine)
    FP_i = calculate_FP_contribution(con_x_cp, actual_class)[0]
    FP_con = calculate_FP_contribution(con_x_cp, actual_class)[1]
    FN_i = calculate_FN_contribution(con_x_cp, actual_class)[0]
    FN_con = calculate_FN_contribution(con_x_cp, actual_class)[1]
    TP_i = calculate_TP(actual_class, con_x_cp)
    P_R_F = calculate_P_R_F(actual_class, TP_i, FP_i, FP_con, FN_i, FN_con)[0]
    results = calculate_P_R_F(actual_class, TP_i, FP_i, FP_con, FN_i, FN_con)[1]
    put_in_dataframe = create_dataframe(P_R_F, results)




if __name__ == "__main__":
    main()