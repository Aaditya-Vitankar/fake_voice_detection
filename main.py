from Resources.lib import *
from Resources import lib as lib
import logging

log_file = "./logs/main.log"

log_file_main = "./logs/web_app.log"

def main(file):
    lg_info = lib.setup_logger("MAIN", log_file, level=logging.INFO)

    lg_info_main = lib.setup_logger("WEB_APP", log_file_main, level=logging.INFO)

    lg_err = lib.setup_logger("MAIN",log_file , level=lib.logging.ERROR)

    lg_err_main = lib.setup_logger("WEB_APP",log_file_main , level=lib.logging.ERROR)
    
    try:
        lg_info.info("Features extraction CALLED")

        if file[-4 :] != ".mp3" and file[-4 :] != ".wav":
            os.remove(file)
            lg_info.info("Unsupported File Format Found")
            lg_info.info(f"File '{file}' deleted successfully.")
            return "unsupported Format"

        if file[-4:] == ".mp3":
            mp3_file = file[:-4] + ".wav"
            convert_mp3_to_wav(file , mp3_file)
        elif file[-4:] == ".wav":
            mp3_file = file
            


        data = create_DataFrame(mp3_file , segment_length=1)
        if len(data) < 7:
            print(data)
            lg_info.info("File Length Smaller than 'Seven' seconds")
            del data
            del mp3_file
            del file
            return "File Length is Small"
    except Exception as e:
         lg_err.error(f"Features extraction FAILED: {e}")

    # try:
    #     lg_info.info("Data reshape CALLED")
    #     reshaped_data = reshape_data(data)
    # except Exception as e:
    #     lg_err.error(f"Data reshape FAILED: {e}")

    try:
        lg_info.info("Classification STARTED")
        
        result = lib.TF_Predict(data)
        
        # result = Dummy_predict(reshaped_data)
        
        # result = lib.torch_predict_LSTM(data) 
        # result = lib.torch_predict_GRU(data,weight=2)
        # result = lib.torch_predict_RNN(data,weight=2)
        
        del data
        del mp3_file
        del file
        lg_info.info("Classification SUCCESSFUL")
        lg_info_main.info("Classification SUCCESSFUL")
        lg_info_main.info("*-"*70)
        lg_info.info("*-"*70)
        return result
    except Exception as e:
        lg_err.error(f"Classification FAILED: {e}")
        lg_err_main.error(f"Classification FAILED: {e}")
        return ""
    
