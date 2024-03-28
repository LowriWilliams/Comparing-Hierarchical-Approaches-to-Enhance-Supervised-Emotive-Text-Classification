import csv
import collections

# Read csv files. Annotations: IP, sentence_id, annotation

def read_file():  
    with open('annotations.csv', 'r') as f:
        reader = csv.reader(f)
        annotations = list(reader)
    
    return annotations


def calculate_majority(annotations):
    all_annotations = collections.defaultdict(list)

    for a, b, c in annotations:
        all_annotations[b].append(c)
    
    counts = []

    for a, b in all_annotations.items():
        for i in b:
            emotion_count = b.count(i)
            tmp = [a, i, emotion_count]
            if tmp not in counts:
                counts.append(tmp)
    
    all_counts = collections.defaultdict(list)

    for a, b, c in counts:
        tmp = [b, c]
        all_counts[a].append(tmp)

    count = 0

    for a, b in all_counts.items():
        for i in b:
            if i[1] >= 3:
                count += 1
                tmp = [i[0]]
                del b[0:]
                b.append(tmp) 

    print('A total of', count, 'sentences out of 1,000 (', ((count/1000)*100), '%) had a majority annotation')
    
    return all_counts


def results(all_counts):
    final = []

    for a, b in all_counts.items():
        for i in b:
            tmp = [a, i[0]]
            final.append(tmp)

    with open("majority_voted.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(final)
        print("Results saved to file!")



##########################################################
    

# Call functions
def main():
    annotations = read_file()
    majority = calculate_majority(annotations)
    save_to_file = results(majority)




if __name__ == "__main__":
    main()