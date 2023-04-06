import csv
import os

# Read the first CSV file and get a column
with open('MGG_8GPU_API_Thread.csv', 'r') as file1:
    reader = csv.reader(file1)
    col0 = [row[0] for row in reader]

with open('MGG_8GPU_API_Thread.csv', 'r') as file1:
    reader = csv.reader(file1)
    col1 = [float(row[1]) for row in reader]
    

# print(col0)
# print(col1)

# Read the second CSV file and get a column
with open('MGG_8GPU_API_Warp.csv', 'r') as file2:
    reader = csv.reader(file2)
    col2 = [float(row[1]) for row in reader]

with open('MGG_8GPU_API_Block.csv', 'r') as file2:
    reader = csv.reader(file2)
    col3 = [float(row[1]) for row in reader]
# print(col2)

# Compute the ratio of the values in the two columns
ratio_1 = [col2[i]/col1[i] for i in range(len(col1))]
ratio_2 = [col3[i]/col1[i] for i in range(len(col1))]

# Write the result, along with the original two columns, to a third CSV file
with open('MGG_API_study.csv', 'w', newline='') as result_file:
    writer = csv.writer(result_file)
    writer.writerow(['Norm.Time w.r.t. Thread', 'MGG_Thread', 'MGG_Warp', 'MGG_Block'])
    for i in range(len(col1)):
        writer.writerow([col0[i].rstrip("_beg_pos"), "1.0x", "{:.3f}".format(ratio_1[i]), "{:.3f}".format(ratio_2[i])])

# os.system("mv MGG_WO_WL.csv csvs/")
# os.system("mv MGG_WL.csv csvs/")