# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------

###-----------------
### Import Libraries
###-----------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections.abc import Callable
from typing import Literal

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


from sklearn.utils import shuffle

import tensorflow as tf

# %matplotlib inline


## --------------- for Frature Extraction
import os
import librosa
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------

###----------------
### Some parameters
###----------------

inpDir = '/mnt/5866F9FE66F9DCA6/Puneet CDAC-Practice/CDAC_Project/DATA/KAGGLE'
outDir = '/mnt/5866F9FE66F9DCA6/Puneet CDAC-Practice/CDAC_Project/DATA'
subDir = 'csvs'
audDir = "AUDIO"

RANDOM_STATE = 24 # REMEMBER: to remove at the time of promotion to production
np.random.seed(RANDOM_STATE) # Set Random Seed for reproducible  results

EPOCHS = 11 # number of epochs
ALPHA = 0.001 # learning rate
NUM_SAMPLES = 1280 # How many samples we want to generate 
NOISE = 0.2 # Noise to be introduced in the data
TEST_SIZE = 0.2

# parameters for Matplotlib
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 8),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'
         }

CMAP = 'coolwarm' # plt.cm.Spectral

plt.rcParams.update(params)


# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------

''' to Extract feature from audio params:
        file_path : <abs. path of the file(.wav format recoomended)>
        segment_length : inverl for taking samples
'''
def extract_features(file_path, segment_length):   # Function to extract features from an audio file
    
    try:
        
        y, sr = librosa.load(file_path) 
        #  Loading audio files returns 
        # y[audio time series. Multi-channel is supported]
        # sr[sampling rate of y] Note: Taking default 22050
        # [For more details : https://librosa.org/doc/0.10.1/generated/librosa.load.html]

        
        num_segments = int(np.ceil(len(y) / float(segment_length * sr))) 
        # Calculate the number of segments based on the segment length and audio length
        
        
        features = [] 
        # Initialize a list to store the features for this file

        
        for i in range(num_segments): # Extracting features for each segment
            
            start_frame = i * segment_length * sr   # Calculate start for the current segment
            end_frame = min(len(y), (i + 1) * segment_length * sr)    # Calculate  end frame for the current segment
            # making sure the last frame does not excede the lenght of audio time series

            
            y_segment = y[start_frame:end_frame]# Extract audio for current segment of audio file


            # Extract different features
            chroma_stft = np.mean(librosa.feature.chroma_stft(y=y_segment, sr=sr))
            # For more details : https://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf
            rms = np.mean(librosa.feature.rms(y=y_segment))
            spec_cent = np.mean(librosa.feature.spectral_centroid(y=y_segment, sr=sr))
            spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_segment, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y_segment))
            mfccs = librosa.feature.mfcc(y=y_segment, sr=sr) # n_mfcc=20 by default
            mfccs_mean = np.mean(mfccs, axis=1)
            
            # Append the extracted features to the list
            features.append([chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, *mfccs_mean])

        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------
# Function to create the dataset


'''Creates the seprate dataset for each of the audio file
   with the dir name as a suffix as label
   audio_dir = str : directory in with the file in
   segment_length : inervel for taking samples
   csv_output_path : path in whic you want ot store the csvs
'''



def create_dataset_sep(audio_dir, segment_length , csv_output_path):
    
    labels = os.listdir(audio_dir) # Label for y
    feature_list = []

    # Iterate over all files in the audio_dir
    for label in labels:
        print(f'Processing {label} files...')
        files = os.listdir(os.path.join(audio_dir, label))
        # Wrap the files iterable with tqdm to show the progress bar
        for file in files:
            file_path = os.path.join(audio_dir, label, file)
            # Extract features for the current file
            file_features = extract_features(file_path, segment_length)
            if file_features:
                # Append features of all segments along with the label to the dataset
                for segment_features in file_features:
                    feature_list.append(segment_features + [label])
                    
            # Create a DataFrame with the dataset
            df = pd.DataFrame(feature_list, columns=['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20','LABEL'])
            fn_name = file.rstrip('.wav')
            csv_output_path = f'./DATA/csvs/{fn_name}_{label}.csv'
            df.to_csv(csv_output_path, index=False)

            print(f'Dataset created and saved to {csv_output_path}')
    
    return None

# Function to create the dataset

# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------

'''Creates the one sngle dataset for all audio file
   with the dir name as a suffix as label
   audio_dir = str : directory in with the file in
   segment_length : inervel for taking samples
'''

def create_dataset(audio_dir, segment_length):
    
    labels = os.listdir(audio_dir) # Label for y
    feature_list = []

    # Iterate over all files in the audio_dir
    for label in labels:
        print(f'Processing {label} files...')
        files = os.listdir(os.path.join(audio_dir, label))
        # Wrap the files iterable with tqdm to show the progress bar
        for file in files:
            file_path = os.path.join(audio_dir, label, file)
            # Extract features for the current file
            file_features = extract_features(file_path, segment_length)
            if file_features:
                # Append features of all segments along with the label to the dataset
                for segment_features in file_features:
                    feature_list.append(segment_features + [label])
                    
    # Create a DataFrame with the dataset
    df = pd.DataFrame(feature_list, columns=['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20','LABEL'])
    
    return df


## Model

# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------

# Check GPU
def check_GPU():
    gpus = tf.config.list_physical_devices('GPU')

    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print (len(gpus), 'Phusical GPUs', len(logical_gpus), 'Logical GPUs')
    except:
        print ('invalid device')

# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------
# train_test_split_DL


def split(batch_size = int , test_size = float ,
          X = pd.DataFrame , y = pd.DataFrame , 
          label_y = False , shuffle_data = False):
    
    ## Finding the lenthh of the test set
    test_len = (len(X)*test_size)//batch_size
    test_len = int(test_len * batch_size)

    ## lablel encoding the the dependeable variable
    ## is specified TRUE
    
    if label_y == True:
        lables = {}
        for i in enumerate(y.unique()):
            lables[i[1]] = i[0]

    print(lables)

    y.replace(lables,inplace=True)

    # Shuffling the data is specified TRUE

    if shuffle_data == True:
        train_df = pd.concat([X , y] , axis=1)
        train_df = shuffle(train_df)
        train_df = train_df.reset_index()
        train_df.drop(columns='index',inplace=True)
        X_train = train_df.drop(columns = 'LABEL')
        y_train = train_df['LABEL']

    # Spliting the data in train test split

    X_train = X[:-test_len] 
    y_train = y[:-test_len]
    
    X_test = X[-test_len:]
    y_test = y[-test_len:]
    

    return X_train ,y_train , X_test , y_test  , lables
