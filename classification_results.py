import numpy as np
import matplotlib.pyplot as plt
 
# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
precision = [10, 20, 35, 100, 34]
recall = [10, 4, 2, 2, 31]
f_measure = [3, 5, 2, 43, 53]
  
# The x position of bars
r1 = np.arange(len(precision))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create bars
plt.bar(r1, precision, width = barWidth, color = 'blue', capsize=7, label='Precision')
plt.bar(r2, recall, width = barWidth, color = 'red', capsize=7, label='Recall')
plt.bar(r3, f_measure, width = barWidth, color = 'green', capsize=7, label='F1-score')

# general layout
plt.xticks([r + barWidth for r in range(len(precision))], ['Flat-hierarchical', 'Relational-based', 'Distance-based', 'Depth-based', 'Semantic-based'])
plt.legend()

# Show graphic

print("Graphs are saved!")
plt.savefig('classification_results.png', bbox_inches='tight')
plt.show()