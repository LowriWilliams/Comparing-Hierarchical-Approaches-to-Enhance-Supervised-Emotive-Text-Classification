from __future__ import division
import csv
import itertools
from itertools import groupby, combinations
import collections
import pandas as pd


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


# Map emotion to index

def map_emotion_to_index(file_as_list, emotions):
    emotion_to_indicies = []

    for i in file_as_list:
        temp = [i[0], i[1], i[2]]
        for j in emotions:
            if i[1] == j[0]:
                temp.append(j[1])
            if i[2] == j[0]:
                temp.append(j[1])
                emotion_to_indicies.append(temp)
    
    unique_indicies = []

    for i in emotions:
        if i[1] not in unique_indicies:
            unique_indicies.append(i[1])

    return emotion_to_indicies, unique_indicies          


# Calculate the weight tree

def calculate_weight_tree(unique_indicies):
    index_length = []
    
    for i in unique_indicies:
        tmp = [i]
        length = len(i)
        if length % 2 != 0:
            length = length - 1
        tmp.append(length)
        index_length.append(tmp)
    
    weight = {10 : 0.3, 12 : 0.2, 14 : 0.1, 16 : 0.08, 18 : 0.06, 20 : 0.04, 22 : 0.02}
    
    for i in index_length:
        for key, value in weight.items():
            if i[1] == key:
                i.append(value)
    
    return index_length


# Calculate misclassification degree

def calculate_misclassification(emotion_to_indicies, index_length):
    for i in emotion_to_indicies:
        M = 0
        for l in index_length:
            if i[3] == i[4]:
                M = 0
            else:
                if i[3] == l[0]:
                    M += l[2]
                if i[4] == l[0]:
                    M += l[2]
        i.append(M)

    return emotion_to_indicies
   

# Calculate con_x_cp

def calculate_con_x_cp(emotion_to_indicies):
    dis_theta = 0.5
    
    for i in emotion_to_indicies:
        con_x_cp = 1 - (i[5]/dis_theta)
        rcon_x_cp = min(1, (max(-1, con_x_cp)))
        i.append(rcon_x_cp)
    
    return emotion_to_indicies


# Calculate FP contribution

def calculate_FP_contribution(emotion_to_indicies):
    actual_classes = []

    for i in emotion_to_indicies:
        if i[1] not in actual_classes:
            actual_classes.append(i[1])
        if i[2] not in actual_classes:
            actual_classes.append(i[2])
    
    FP_i_list = []
    FP_con_list = []

    for i in actual_classes:
        FP_i = [i]
        tmp = []
        FP_con_count = 0
        for d in emotion_to_indicies:
            if i == d[1] and i != d[2]:
                if d[2] not in tmp:
                    FP_con_count += d[6]     
                    tmp.append(d[2])
        FP_i.append(tmp)
        
        FP_con = [i, FP_con_count]
        FP_con_list.append(FP_con)
        FP_i_list.append(FP_i)
    
    for i in FP_i_list:
        try:
            count = len(i[1])
            i.append(count)
            del i[1]
        except:
            i.append(0)
    
    return FP_i_list, FP_con_list, actual_classes


# Calculate FN contribution

def calculate_FN_contribution(emotion_to_indicies, actual_classes):    
    FN_i_list = []
    FN_con_list = []

    for i in actual_classes:
        FN_i = [i]
        tmp = []
        FN_con_count = 0
        for d in emotion_to_indicies:
            if i == d[2] and i != d[1]:
                if d[1] not in tmp:
                    FN_con_count += d[6]
                    tmp.append(d[1])
        FN_i.append(tmp)
                    
        FN_con = [i, FN_con_count]
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

def calculate_TP(emotion_to_indicies, actual_classes):    
    TP_i_list = []
    
    for i in actual_classes:
        tmp = []
        TP_i = [i]
        for d in emotion_to_indicies:
            if i == d[1] and i == d[2]:
                if d[1] not in tmp:
                    tmp.append(d[1])
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
            if i == a[0] and i == b[0] and i == c[0] and i == d[0] and i == e[0]:
                p_top = float(max(0, (a[1] + c[1] + e[1])))
                p_bottom = float(a[1] + b[1] + e[1])
                try:
                    p = round(((p_top/p_bottom)*100), 2)
                except:
                    p = 0.0
                p1_tmp.append(p_top)
                p2_tmp.append(p_bottom)

                r_top = (max(0, (a[1] + c[1] + e[1])))
                r_bottom = (a[1] + d[1] + c[1])
                try:
                    r = round(((r_top/r_bottom)*100), 2)
                except:
                    r = 0.0
                r1_tmp.append(r_top)
                r2_tmp.append(r_bottom)

                try:
                    f = round((2 * ((p*r)/(p+r))), 2)
                except:
                    f = 0.0

                tmp = [i, a[1], b[1], c[1], d[1], e[1], p, r, f]
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

    data_frame.to_csv("depth_based_classification_results_j48_testing.csv", sep=',', index=False)
    
    print("\nResults have been saved as a csv file!\n")


            





##########################################################
    

# Call functions
def main():
    file_as_list = read_file()[0]
    emotions = read_file()[1]
    emotion_to_indicies = map_emotion_to_index(file_as_list, emotions)[0]
    unique_indicies = map_emotion_to_index(file_as_list, emotions)[1]
    weight_tree = calculate_weight_tree(unique_indicies)
    misclassification = calculate_misclassification(emotion_to_indicies, weight_tree)
    con_x_cp = calculate_con_x_cp(misclassification)
    FP_i = calculate_FP_contribution(con_x_cp)[0]
    FP_con = calculate_FP_contribution(con_x_cp)[1]
    actual_class = calculate_FP_contribution(con_x_cp)[2]
    FN_i = calculate_FN_contribution(con_x_cp, actual_class)[0]
    FN_con = calculate_FN_contribution(con_x_cp, actual_class)[1]
    TP_i = calculate_TP(con_x_cp, actual_class)
    P_R_F = calculate_P_R_F(actual_class, TP_i, FP_i, FP_con, FN_i, FN_con)[0]
    results = calculate_P_R_F(actual_class, TP_i, FP_i, FP_con, FN_i, FN_con)[1]
    put_in_dataframe = create_dataframe(P_R_F, results)





if __name__ == "__main__":
    main()