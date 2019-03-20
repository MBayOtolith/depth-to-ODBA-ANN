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

""" Main script for training the ANN model to predict ODBA based on dive depth data.

See our method paper published in Animal Biotelemetry:
Liu, Z.Y.-C., Moxley, J.H. et al. (2019) Deep learning accurately predicts white shark 
locomotor activity from depth data.

Data provided to reproduce the results: shark_1_data.csv and shark_2_data.csv
Data column: time (sec), depth (m), ODBA
Each dataset has 25,200 data points, 1 data point is 1 second measurement.
First hour of data are used for training.
The following 6 hours of data are for validation (outside/dev data)

Utility scripts:
smooth.py: Function to smooth the data
window_size.py: Function to apply moving window size to data
ANN_train.py: data processing, ANN architecture and model building

After running the main script, the R2 metrics will be outputted to a csv file,
and the plots of training and prediction will be generated.

For questions, email: zacqoo@gmail.com 
"""

# Import libraries
import pandas as pd
from ANN_train import ANN_train

# Load dataset from csv file
df = pd.read_csv('shark_1_data.csv')
#df = pd.read_csv('shark_2_data.csv')

# Set up model parameters
start_point = 0 
end_point = 3600 # only use 1 hours data (3600 data points) for training
win_1= 20 # moving window size
batch_s = 64 # batch size
epoch_n = 100 # number of training epochs

# Run the training, output performance metrics and plots
for i in range(0, 1):
    ANN_model_training_points(df, start_point, end_point, win_1, batch_s, epoch_n)

