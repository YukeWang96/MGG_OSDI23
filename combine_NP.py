# import pandas as pd

# # Read the first CSV file and get a column
# df1 = pd.read_csv('MGG_8GPU_WO_NP.csv', header=None)
# data_li = df1.iloc[:, 0] # use iloc to get the first (and only) column
# UVM_time = df1.iloc[:, 1] # use iloc to get the first (and only) column

# # Read the second CSV file and get a column
# df2 = pd.read_csv('MGG_8GPU_NP.csv', header=None)
# MGG_time = df2.iloc[:, 1] # use iloc to get the first (and only) column

# # Compute the ratio of the values in the two columns
# ratio = UVM_time / MGG_time

# # Create a new DataFrame with the original columns and the ratio column
# result = pd.DataFrame({ 'data': data_li,
#                         'MGG_WO_NP': UVM_time,
#                         'MGG_NP': MGG_time,
#                         'Speedup (x)': ratio })

# # Write the result to a third CSV file
# result.to_csv('result.csv', index=False)


import csv

# Read the first CSV file and get a column
with open('MGG_8GPU_WO_NP.csv', 'r') as file1:
    reader = csv.reader(file1)
    col0 = [row[0] for row in reader]

with open('MGG_8GPU_WO_NP.csv', 'r') as file1:
    reader = csv.reader(file1)
    col1 = [float(row[1]) for row in reader]

print(col0)
print(col1)

# Read the second CSV file and get a column
with open('MGG_8GPU_NP.csv', 'r') as file2:
    reader = csv.reader(file2)
    col2 = [float(row[1]) for row in reader]
print(col2)

# Compute the ratio of the values in the two columns
ratio = [col1[i] / col2[i] for i in range(len(col1))]

# Write the result, along with the original two columns, to a third CSV file
with open('result.csv', 'w', newline='') as result_file:
    writer = csv.writer(result_file)
    writer.writerow(['Dataset', 'MGG_WO_NP', 'MGG_W_NP', 'Speedup (x)'])
    for i in range(len(col1)):
        writer.writerow([col0[i].rstrip("_beg_pos"), col1[i], col2[i], "{:.3f}".format(ratio[i])])