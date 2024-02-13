Simply put the weights and dataframe object as shown in below sample function calling

Applicable for all the functions provide in the folder

Weights are provide in the same folder, GRU have 3 weights, RNN have 2 weights and LSTM have 1 weight, you can use any one at a time.

Weights should match with the corresponding build model inside the functions.

You can make necessary modification if required 

predicted_label = predict_label(weights, df) sample function calling

print("Predicted label:", predicted_label)
