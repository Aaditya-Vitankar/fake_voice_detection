# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------

# Logger

"""To set up as many loggers as you want"""
import logging
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
def setup_logger(name, log_file, level=logging.WARNING):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        filemode='w')

    logger = logging.getLogger(name)
    if logger.handlers:
        logger.handlers = []
    logger.addHandler(handler)


    return logger


# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------
import os

log_file = "./logs/main.log"

if not os.path.exists("./logs/"):
    os.makedirs("./logs/")
    with open("./logs/main.log",'w'):
        pass

lg_info = setup_logger("LIB",log_file , level=logging.INFO)

lg_err = setup_logger("LIB",log_file, level=logging.ERROR)

lg_war = setup_logger("LIB",log_file , level=logging.WARNING)


###-----------------
### Import Libraries
###-----------------

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# TensorFlow Imports
# import tensorflow as tf
from Resources.model_package import *
from Resources import model_package as MODEL

# PyTorch Imports
# from Resources.torch_models import LSTM_Prediction_Function as model
# from Resources.torch_models.LSTM_Prediction_Function import *

# from Resources.torch_models import GRU_Prediction_Function as model
# from Resources.torch_models.GRU_Prediction_Function import *

# from Resources.torch_models import RNN_Prediction_Function as model
# from Resources.torch_models.RNN_Prediction_Function import *

# %matplotlib inline

# for Frature Extraction
import os
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment


# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------


''' to Extract feature from audio params:
        file_path : <abs. path of the file(.wav format recoomended)>
        segment_length : inverl for taking samples
'''
def extract_features(file_path, segment_length):   # Function to extract features from an audio file
    lg_info.info("Feature Extraction STARTED")
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
        lg_info.info("Feature Extraction SUCCESSFUL")
        return features
    
    except Exception as e:
        lg_err.error(f" Feature Extraction FAILED: {e}")
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
        
        lg_info.info("Dataset creation SUCCESSFUL")
        return df
    except Exception as e:
        lg_err.error(f"Dataset creation FAILED: {e}")



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
        lg_info.info("GPU Configration SUCCESSFUL")
    except:
        lg_err.error("GPU Configration FAILED")
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
        lg_info.info("DataFrame creation SUCCESSFUL")
        return df
    except Exception as e:
        lg_err.error(f"DataFrame creation FAILED: {e}")

# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------


def convert_mp3_to_wav(mp3_file, wav_file):
    try:
        # Load the MP3 file
        audio = AudioSegment.from_mp3(mp3_file)
        
        # Export the audio to WAV
        audio.export(wav_file, format="wav")
        
        print(f"MP3 file '{mp3_file}' converted to WAV file '{wav_file}' successfully.")
    except Exception as e:
        print(f"Error converting MP3 to WAV: {e}")



# Function to create the dataset

'''Creates the seprate dataset for each of the audio file
   with the dir name as a suffix as label
   audio_dir = str : directory in with the file in
   segment_length : inervel for taking samples
   csv_output_path : path in whic you want ot store the csvs
'''

def create_dataset_sep(audio_dir, segment_length , csv_output_path):
    try:
        lg_info.info("DataFrame creation STARED")
        
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
                lg_info.info(f"Dataset created and saved to {csv_output_path+fn_name}.csv")
        
        lg_info.info("DataFrame creation SUCCESSFUL")
        return None
    except Exception as e:
        lg_err.error(f"DataFrame creation FAILED: {e}")

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

def TF_Predict(data = pd.DataFrame):
    model = MODEL.Model()
    model.buid_model()
    model.load_weights("Weights/demo_weigths")
    result = model.predict(data)
    return result

# ---------------------------------------------------------------------------------------------
# =============================================================================================
# ---------------------------------------------------------------------------------------------

# def torch_predict_GRU(data = pd.DataFrame , weight = 1):
#     """PyTorch Prediction Funtion with GRU Model
#         data   : Pandas DataFrame : 26 Columns
#         Weight : Pretrained Weight Selection : takes integer pass in between 1 and 3 """
    
#     if weight not in [1,2,3]:
#         return "Invalid weight"
#     else:
#         weight = weight-1

#         wt =["./Weights/torch/GRU1/DeepfakeGRU.pt",
#             "./Weights/torch/GRU2/DeepfakeGRU.pt",
#             "./Weights/torch/GRU3/DeepfakeGRU.pt"]
#         return model.predict_label(weights=wt[weight] , input_data=data)

# # ---------------------------------------------------------------------------------------------
# # =============================================================================================
# # ---------------------------------------------------------------------------------------------    

# def torch_predict_RNN(data = pd.DataFrame, weight = 1):
#     """PyTorch Prediction Funtion with RNN Model
#         data   : Pandas DataFrame : 26 Columns
#         Weight : Pretrained Weight Selection : takes integer input as 1 or 2 """
    
#     if weight not in [1 ,2]:
#         return "Invalid weight"
#     else:
#         weight = weight-1
#         wt = ["./Weights/torch/RNN1/DeepFakeRNN.pt",
#         "./Weights/torch/RNN2/DeepFakeRNN.pt"]
#         return model.predict_label(weights=wt[weight] , input_data=data)

# # ---------------------------------------------------------------------------------------------
# # =============================================================================================
# # --------------------------------------------------------------------------------------------- 

# def torch_predict_LSTM(data = pd.DataFrame):
#     """PyTorch Prediction Funtion with RNN Model
#         data   : Pandas DataFrame : 26 Columns"""
    
#     wt = ["./Weights/torch/LSTM/DeepFakeLSTM.pt"]
#     return model.predict_label(weights=wt[0] , input_data=data)
    
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

            lg_info.info("Data Reshape with lables SUCCESSFUL")
            return new_data , y_re
        else:
            lg_info.info("Data Reshaping SUCCESSFUL")
            return new_data
    
    except Exception as e:
        lg_err.error(f" Data Reshaping FAILED: {e}")
        