import torch,torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Define the RNN model
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size,batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
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

        # Extract the probability corresponding to the predicted label
        predicted_label = torch.argmax(output).item()
        probability_of_accuracy = probabilities[0][predicted_label].item()

        if predicted_label == 0:
            return f"This audio is FAKE with {probability_of_accuracy*100:.2f}% probability of accuracy"
        else:
            return f"This audio is REAL with {probability_of_accuracy*100:.2f}% probability of accuracy"

# predicted_label = predict_label(weights, df) function calling
#print("Predicted label:", predicted_label)