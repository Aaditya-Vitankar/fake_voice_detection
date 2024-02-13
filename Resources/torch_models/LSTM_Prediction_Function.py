import torch,torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:,-1,:])
        return out
    

# weights = "weights/Gru2/DeepfakeGRU.pt" example weights
def preprocess_data(input_data):
    """
    Preprocess the input data.

    Args:
        data (pd.DataFrame): Input data in the form of a pandas DataFrame.

    Returns:
        torch.Tensor: Preprocessed input data.
    """
    # Perform any necessary preprocessing, such as scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_data)
    return torch.tensor(scaled_data, dtype=torch.float32)

def predict_label(weights, input_data):
    """
    Predict label using the trained LSTM model and input data.

    Args:
        model (torch.nn.Module): The trained LSTM model.
        input_data (pd.DataFrame): Input data in the form of a pandas DataFrame.

    Returns:
        int: Predicted label.
    """
    model = torch.load(weights)
    # Preprocess the input data
    input_tensor = preprocess_data(input_data)

    # # Reshape input tensor if necessary
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Set the model to evaluation mode
    model.eval()

    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor)

        probabilities = F.softmax(output,dim=1)

        final_proba = probabilities.cpu().detach().numpy().tolist()[0]

        if 1.00 > final_proba[0] > 0.8:
            return f'Audio is Real'
        
        elif 0.8 > final_proba[0] > 0.5:
            return f'Audio Seems Real with {final_proba[0] * 100: .2f}% Probability'
        
        elif 0.5 > final_proba[0] > 0.2:
            return f'Audio Seems Fake with {final_proba[1] * 100: .2f}% Probability'
        
        elif 0.2 > final_proba[0] > 0:
            return 'Audio is Fake'

