import pandas as pd

# Read the first CSV file and get a column
df1 = pd.read_csv('1_dgl_gcn.csv', header=None)
data_li = df1.iloc[:, 0] # use iloc to get the first (and only) column
UVM_time = df1.iloc[:, 1] # use iloc to get the first (and only) column

# Read the second CSV file and get a column
df2 = pd.read_csv('file2.csv', header=None)
MGG_time = df2.iloc[:, 1] # use iloc to get the first (and only) column

# Compute the ratio of the values in the two columns
ratio = UVM_time / MGG_time

# Create a new DataFrame with the original columns and the ratio column
result = pd.DataFrame({ 'data': data_li,
                        'UVM_t': UVM_time,
                        'MGG_t': MGG_time,
                        'Speedup (x)': ratio })

# Write the result to a third CSV file
result.to_csv('result.csv', index=False)