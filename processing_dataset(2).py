import numpy as np

def import_data(file_path):
    data_target = []
    data_input = []
    with open(file_path, 'r') as f:  #open file Querylevelnorm ,r means read only
        while True:
            # Splitting our dataset per colums based on the space separator
            data_str = f.readline().split(' ')
            if not data_str[0]:  #non0 means true,if not true means it couldn't be executed
                break
            # Target is the first column
            data_target.append(float(data_str[0]))  #add data_str[0] to data_target
            # Features are columns 2 to 48, where we take only the values
            data_input.append([float(a.split(':')[1]) for a in data_str[2:48]])
            #add a from[2:48]colums to data_input 
    x = np.array(data_input)
    y = np.array(data_target).reshape((-1, 1))  #-1 means arrange data in a new way
    #1 means 1 colum ,if(2,-1) 2 means 2 rows 
    #np.arange(0,10)  // generate [0,1,2,3,4,5,6,7,8,9]
    #d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float)
    #np.array save single datatype multidimensional array ,from python list/tuple
    #arange could generate float type,but range generates int type,
    return x, y

#file_path = r'path\to\file\Querylevelnorm.txt' 
file_path = r'C:\Users\Administrator\Desktop\Machine Learning\Project2\Querylevelnorm.txt'
x, y = import_data(file_path)
print("X shape is", x.shape, "\nY shape is", y.shape)

n, m = x.shape  #x's dimensions and the number of each array's elements

# Data partition
n_train = int(0.8 * n)  #80% datapoints
n_valid = int((n - n_train) / 2)  #10% datapoints
print("n_train is",n_train,"\nn_valid is",n_valid)

x_train = x[:n_train, :]  #0 to n_train rows80%
x_valid = x[n_train:n_train + n_valid, :]  #n_train to n-valid
x_test = x[n_train + n_valid:, :]  #n_valid to end
print("x_train is",x_train,"x_valid is",x_valid,"x_test is",x_test)

y_train = y[:n_train, :]
y_valid = y[n_train:n_train + n_valid, :]
y_test = y[n_train + n_valid:, :]
print("y_train is",y_train,"y_valid is",y_valid,"y_test is",y_test)