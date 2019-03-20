# MIT License

# Copyright (c) 2019 Otolith- Monterey Bay Aquarium- Conservation Research

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""
Utility script for ANN model training setup.

1. process the data
2. set up model parameters: moving window size
3. set up ANN architecture using Keras
4. make predictions for the next 6 hours of data points (outside/ dev data)
5. calculate metrics (R2)
6. plots

"""

def ANN_train(df, i_start, i_end, win_1, batch_s, epoch_n):
    import numpy as np
    
    ### 1. process the data ----------------------------------------------------
    data = df.iloc[:, [0, 1, 2]].values # time, depth, ODBA
    # stack depth into one dataset
    dataset = np.vstack(data[:,1])
    
    ### 2. set up model parameters: moving window size  ------------------------
    # Setup parameters
    row = i_end - i_start
    feature_n = win_1
    L1 = win_1 + 10 # ANN layer 1 number of neurons
    L2 = win_1 + 10 # ANN layer 2 number of neurons
    L3 = win_1 + 10 # ANN layer 3 number of neurons
    
    # Create training set with moving window
    from window_size import mv_window
    X = mv_window(row, feature_n, i_start, win_1, dataset)
    # Create testing set
    from smooth import sm
    y = sm(data[i_start:i_end,2],30)
    
    # Setup parameters for outside dataset/ dev set
    i_start_a = i_end
    i_end_a = i_start_a + 21600
    row_a = i_end_a - i_start_a
    # Create training set with moving window
    from window_size import mv_window
    X_a = mv_window(row_a, feature_n, i_start_a, win_1, dataset)
    # Create testing set
    from smooth import sm
    y_a = sm(data[i_start_a:i_end_a,2],30)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    #X = sc.fit_transform(X)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_all = sc.transform(X)
    X_all_a = sc.transform(X_a)

    ### 3. set up ANN architecture using Keras  -------------------------------
    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    #from keras.layers import Dropout
    # Building ANN
    regressor = Sequential()
    regressor.add(Dense(units = L1, kernel_initializer = 'normal', activation = 'relu', input_dim = feature_n))
    regressor.add(Dense(units = L2, kernel_initializer = 'normal', activation = 'relu'))
    regressor.add(Dense(units = L3, kernel_initializer = 'normal', activation = 'relu'))
    regressor.add(Dense(units = 1, kernel_initializer = 'normal'))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    ### 4. make predictions for the next 6 hours of data points (outside/ dev data) ------------
    # Fit the ANN to training set
    regressor.fit(X_train, y_train, batch_size = batch_s, epochs = epoch_n)
    #regressor.load_weights(weights_61200)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Predicting the whole dataset
    y_pred_all = regressor.predict(X_all)
    # Predicting outside dataset/ dev set
    y_pred_all_a = regressor.predict(X_all_a)

    ### save weights
    #regressor.save_weights("weights_" + str(i_start))

    ### 5. calculate metrics (R2) and output them -------------------------------------------
    # Calculate rmse and r^2
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    r_score = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) 
    #r_score_all = r2_score(y, y_pred_all) #0.4721
    #rmse_all = mean_squared_error(y, y_pred_all) #0.2935
    r_score_a = r2_score(y_a, y_pred_all_a) #0.4721
    rmse_a = mean_squared_error(y_a, y_pred_all_a) #0.2935
    # 1st hour of Test set
    r_score_1 = r2_score(y_a[0:3600], y_pred_all_a[0:3600,0])
    rmse_1 = mean_squared_error(y_a[0:3600], y_pred_all_a[0:3600,0])
    # 2nd hour of Test set
    r_score_2 = r2_score(y_a[3600:7200], y_pred_all_a[3600:7200,0])
    rmse_2 = mean_squared_error(y_a[3600:7200], y_pred_all_a[3600:7200,0])
    # 3rd hour of Test set
    r_score_3 = r2_score(y_a[7200:10800], y_pred_all_a[7200:10800,0])
    rmse_3 = mean_squared_error(y_a[7200:10800], y_pred_all_a[7200:10800,0])
    # 4th hour of Test set
    r_score_4 = r2_score(y_a[10800:14400], y_pred_all_a[10800:14400,0])
    rmse_4 = mean_squared_error(y_a[10800:14400], y_pred_all_a[10800:14400,0])
    # 5th hour of Test set
    r_score_5 = r2_score(y_a[14400:18000], y_pred_all_a[14400:18000,0])
    rmse_5 = mean_squared_error(y_a[14400:18000], y_pred_all_a[14400:18000,0])
    # 6th hour of Test set
    r_score_6 = r2_score(y_a[18000:21600], y_pred_all_a[18000:21600,0])
    rmse_6 = mean_squared_error(y_a[18000:21600], y_pred_all_a[18000:21600,0])
    # output all metric scores of R2 and rmse
    import csv
    csvRow = [i_start,(i_start-i_end)/3600,r_score,rmse,r_score_a,rmse_a,
        r_score_1,rmse_1,r_score_2,rmse_2,r_score_3,rmse_3,r_score_4,rmse_4,r_score_5,rmse_5,r_score_6,rmse_6]
    
    csvfile = "output_stats_ts.csv"
    with open(csvfile, "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(csvRow)
    # save outputs and predictions
    np.savetxt("y_training" + str(i_start), y, delimiter=",")
    np.savetxt("y_training_pred_" + str(i_start), y_pred_all, delimiter=",")
    np.savetxt("y_a_" + str(i_start), y_a, delimiter=",")
    np.savetxt("y_pred_" + str(i_start), y_pred_all_a, delimiter=",")

    ### 6. plots  -------------------------------------------------------------
    # create and save figures
    from smooth import sm
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,5))
    plt.gca().set_color_cycle(['blue','red'])
    plt.plot(data[0:3600,0], sm(y[0:3600],30))
    plt.plot(data[0:3600,0], sm(y_pred_all[0:3600,0],30))
    plt.ylabel('ODBA')
    plt.xlabel('Time- 1 hours period')
    plt.legend(['Data', 'Prediction'], loc='upper right')
    #plt.show()
    plt.savefig("Fig_train_1_" + str(i_start) +".png", format="PNG")

    # For outside dataset: 0-1200 sec
    plt.figure(figsize=(15,5))
    plt.gca().set_color_cycle(['blue','red'])
    plt.plot(data[i_end:i_end+7200,0], sm(y_a[0:7200],30))
    plt.plot(data[i_end:i_end+7200,0], sm(y_pred_all_a[0:7200,0],30))
    plt.ylabel('ODBA')
    plt.xlabel('Time- 2 hours period')
    plt.legend(['Data', 'Prediction'], loc='upper right')
    #plt.show()
    plt.savefig("Fig_test_1_" + str(i_start) +".png", format="PNG")
    # For outside dataset: 1200-2400 sec
    plt.figure(figsize=(15,5))
    plt.gca().set_color_cycle(['blue','red'])
    plt.plot(data[i_end+7200:i_end+14400,0], sm(y_a[7200:14400],30))
    plt.plot(data[i_end+7200:i_end+14400,0], sm(y_pred_all_a[7200:14400,0],30))
    plt.ylabel('ODBA')
    plt.xlabel('Time- 2 hours period')
    plt.legend(['Data', 'Prediction'], loc='upper right')
    #plt.show()
    plt.savefig("Fig_test_2_" + str(i_start) +".png", format="PNG")
    # For outside dataset: 2400-3600 sec
    plt.figure(figsize=(15,5))
    plt.gca().set_color_cycle(['blue','red'])
    plt.plot(data[i_end+14400:i_end+21600,0], sm(y_a[14400:21600],30))
    plt.plot(data[i_end+14400:i_end+21600,0], sm(y_pred_all_a[14400:21600,0],30))
    plt.ylabel('ODBA')
    plt.xlabel('Time- 2 hours period')
    plt.legend(['Data', 'Prediction'], loc='upper right')
    #plt.show()
    plt.savefig("Fig_test_3_" + str(i_start) +".png", format="PNG")
    
    # For outside dataset: 0-3600 sec
    plt.figure(figsize=(15,5))
    plt.gca().set_color_cycle(['blue','red'])
    plt.plot(data[i_start_a:i_end_a,0], sm(y_a,30))
    plt.plot(data[i_start_a:i_end_a,0], sm(y_pred_all_a[:,0],30))
    plt.ylabel('ODBA')
    plt.xlabel('Time- 6 hours period')
    plt.legend(['Data', 'Prediction'], loc='upper right')
    #plt.show()
    plt.savefig("Fig_test_4_" + str(i_start) +".png", format="PNG")
    
    return