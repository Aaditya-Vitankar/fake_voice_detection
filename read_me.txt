from model_package import Model

model = Model()
model.buid_model()
model.load_weights('demo/demo_weigths')
model.predict() # Just pass the dataframe in the predict function.