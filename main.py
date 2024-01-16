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
        data = create_DataFrame(file , segment_length=1)
    except:
         lg_err.error(f"Features extraction FAILED: {e}")

    try:
        lg_info.info("Data reshape CALLED")
        reshaped_data = reshape_data(data)
    except Exception as e:
        lg_err.error(f"Data reshape FAILED: {e}")

    try:
        lg_info.info("Classification STARTED")
        result = Dummy_predict(reshaped_data)
        lg_info.info("Classification SUCCESSFUL")
        lg_info_main.info("Classification SUCCESSFUL")
        lg_info_main.info("*-"*70)
        lg_info.info("*-"*70)
        return result
    except Exception as e:
        lg_err.error(f"Classification FAILED: {e}")
        lg_err_main.error(f"Classification FAILED: {e}")
        return ""
    
