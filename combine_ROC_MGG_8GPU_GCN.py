import csv
import os

# Read the first CSV file and get a column
# with open('UVM_GCN_8GPU.csv', 'r') as file1:
#     reader = csv.reader(file1)
#     col0 = [row[0] for row in reader]

# check the existance of those .csv file
if not os.path.exists("roc-new/3_run_d16_it100.csv"):
    print("ROC GCN has not been executed yet!")
    exit(-1)

if not os.path.exists("MGG_GCN_8GPU.csv"):
    print("MGG GCN has not been executed yet!")
    exit(-1)


with open('roc-new/3_run_d16_it100.csv', 'r') as file1:
    reader = csv.reader(file1)
    col1 = []
    for row in reader:
        try:
            col1.append(float(row[1]))
        except:
            continue
    # col1 = [float(row[1]) for row in reader]
    
# print(col0)
# print(col1)

# Read the second CSV file and get a column
with open('MGG_GCN_8GPU.csv', 'r') as file2:
    reader = csv.reader(file2)
    col2 = [float(row[1]) for row in reader]

with open('MGG_GCN_8GPU.csv', 'r') as file1:
    reader = csv.reader(file1)
    col0 = [row[0] for row in reader]
# print(col2)

# Compute the ratio of the values in the two columns
ratio = [col1[i] / col2[i] for i in range(len(col1))]

# Write the result, along with the original two columns, to a third CSV file
with open('Fig_9_ROC_MGG_GCN_8GPU.csv', 'w', newline='') as result_file:
    writer = csv.writer(result_file)
    writer.writerow(['Dataset', 'ROC-8GPU(ms)', 'MGG-8GPU(ms)', 'Speedup (x)'])
    for i in range(len(col1)):
        writer.writerow([col0[i].rstrip("_beg_pos.bin"), col1[i], col2[i], "{:.3f}".format(ratio[i])])

# os.system("mv UVM_8GPU.csv csvs/")
# os.system("mv MGG_8GPU.csv csvs/")