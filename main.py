from Resources.lib import *
from Resources import lib as lib

def main(file):
    lg_info = lib.setup_logger("MAIN","./logs/backend.log" , level=lib.logging.INFO)

    lg_err = lib.setup_logger("MAIN","./logs/backend.log" , level=lib.logging.ERROR)

    lg_war = lib.setup_logger("MAIN","./logs/backend.log" , level=logging.WARNING)
    
    try:
        data = create_DataFrame(file , segment_length=1)
        lg_info.info("INFO: Features extraction SUCCESSFUL")
    except:
         lg_err.error(f"INFO: Features extraction FAILED: {e}")

    try:
        reshaped_data = reshape_data(data)
        lg_info.info("INFO: Data reshape SUCCESSFUL")
    except Exception as e:
        lg_err.error(f"ERROR: Data reshape FAILED: {e}")

    try:
        result = Dummy_predict(reshaped_data)
        lg_info.info("INFO: Classification SUCCESSFUL")
        return result
    except Exception as e:
        lg_err.error(f"ERROR: Classification FAILED: {e}")
    
