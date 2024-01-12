# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------

# Logger

'''Use to Keep Logs'''
import logging
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
def setup_logger(name, log_file, level=logging.INFO):
    """To set up as many loggers as you want"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        logger.handlers = []
    logger.addHandler(handler)

    return logger


# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------

lg_info = setup_logger("LIB","./logs/lib.log" , level=logging.INFO)

lg_err = setup_logger("LIB","./logs/lib.log" , level=logging.ERROR)

lg_war = setup_logger("LIB","./logs/lib.log" , level=logging.WARNING)


###-----------------
### Import Libraries
###-----------------
lg_info.info("INFO:  Import STARTED")
try:
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

    import tensorflow as tf

    from sklearn.utils import shuffle

    # import torch
    # import torch.nn as nn
    # import torch.nn.functional as F
    # from torch.autograd import Variable

    # %matplotlib inline

    # for Frature Extraction
    import os
    import librosa
    import numpy as np
    import pandas as pd
    from tqdm.notebook import tqdm
    lg_info.info("INFO: Import SUCCESSFUL")
except Exception as e:
    lg_err.error(f"ERROR: Import FAILED: {e}")



# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------


''' to Extract feature from audio params:
        file_path : <abs. path of the file(.wav format recoomended)>
        segment_length : inverl for taking samples
'''
def extract_features(file_path, segment_length):   # Function to extract features from an audio file
    lg_info.info("INFO: Feature Extraction STARTED")
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
        lg_info.info("INFO: Feature Extraction SUCCESSFUL")
        return features
    
    except Exception as e:
        lg_err.error(f"ERROR: Feature Extraction FAILED: {e}")
        print(f"Error processing {file_path}: {e}")
        return None


# Function to create the dataset

'''Creates the one sngle dataset for all audio file
   with the dir name as a suffix as label
   audio_dir = str : directory in with the file in
   segment_length : inervel for taking samples
'''

def create_dataset(audio_dir, segment_length):
    
    try:
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
        
        lg_info.info("INFO: Dataset creation SUCCESSFUL")
        return df
    except Exception as e:
        lg_err.error(f"INFO: Dataset creation FAILED: {e}")



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
        lg_info.info("INFO: GPU Configration SUCCESSFUL")
    except:
        lg_err.error("INFO: GPU Configration FAILED")
        print ('invalid device')

# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------


# Function to create the DataFrame

'''Extracts Fetures from Audio and  returns DataFrame (Features Only)
        File_path : Audio File Path
        Segment_length : Length of Segment you want to extract fetures'''

def create_DataFrame(File_path, segment_length):
    
    try:
        feature_list =[]
        file_features = extract_features(File_path, segment_length)
        if file_features:
            # Append features of all segments along with the label to the dataset
            for segment_features in file_features:
                feature_list.append(segment_features)
                        
        # Create a DataFrame with the dataset
        df = pd.DataFrame(feature_list, columns=['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'])
        lg_info.info("INFO: DataFrame creation SUCCESSFUL")
        return df
    except Exception as e:
        lg_err.error(f"INFO: DataFrame creation FAILED: {e}")

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
    try:
        lg_info.info("INFO: DataFrame creation STARED")
        
        labels = os.listdir(audio_dir) # Label for y
        

        # Iterate over all files in the audio_dir
        for label in labels:
            print(f'Processing {label} files...')
            files = os.listdir(os.path.join(audio_dir, label))
            # Wrap the files iterable with tqdm to show the progress bar
            for file in files:
                file_path = os.path.join(audio_dir, label, file)

                df = create_DataFrame(file_path, segment_length)

                df['LABEL'] = [label for _ in range(len(df))]
                fn_name = file.rstrip('.wav')
                
                df.to_csv(csv_output_path+fn_name+".csv", index=False)

                print(f'Dataset created and saved to {csv_output_path+fn_name+".csv"}')
                lg_info.info(f"INFO: Dataset created and saved to {csv_output_path+fn_name}.csv")
        
        lg_info.info("INFO: DataFrame creation SUCCESSFUL")
        return None
    except Exception as e:
        lg_err.error(f"INFO: DataFrame creation FAILED: {e}")

# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------

'''use to Simulate AI Model when AI model is not Avilable (ONLY FOR DEVLOPMENT PERPOSE)'''

def Dummy_predict(data = np.array):
    if data.shape[2] == 26:
        return np.random.choice(['REAL','FAKE'])
    else:
        return 'FAKE'

# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------
    
def reshape_data(data = pd.DataFrame ,label = 'NONE', time_step = 30 , time_interval = 10):

    try:
        rem = len(data)% time_step
        if rem!=0:
            data = data.iloc[:-rem]
        x_dim = data.shape[1]
        y_dim = 1
        z_dim = time_step
        new_data = data.iloc[:time_step].to_numpy().reshape(y_dim,z_dim,x_dim)

        for i in range(time_step,len(data)-time_step , time_interval):
            part = data.iloc[i:i+time_step]
            part = part.to_numpy().reshape(y_dim,z_dim,x_dim)
            new_data = np.concatenate((new_data, part),axis=0)
        
        if label != 'NONE':
            y_re = [label for _ in range(len(new_data))]
            len(y_re)

            lg_info.info("INFO: Data Reshape with lables SUCCESSFUL")
            return new_data , y_re
        else:
            lg_info.info("INFO: Data Reshaping SUCCESSFUL")
            return new_data
    
    except Exception as e:
        lg_err.error(f"ERROR: Data Reshaping FAILED: {e}")
        