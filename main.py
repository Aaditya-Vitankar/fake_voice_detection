from Resources.lib import *
from Resources import lib as lib

def main(file):
    data = create_DataFrame(file , segment_length=1)
    reshaped_data = reshape_data(data)
    return Dummy_predict(reshaped_data)
