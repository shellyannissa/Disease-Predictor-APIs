  # Overview
This is a collection of APIs written in python with the purpose of discerning diseases based on symptoms, confirming their presence, and predicting potential occurrences.Various regression models and machine learning algorithms have been utilised to fit the data and predict upon it. 

Among the models,is one that predicts the disease diagnosis based on a set of entered symptoms, chosen from a predetermined list of 41 common symptoms. Additionally, there are eight advanced disease predictor models that assess whether an individual has a particular disease based on a complex set of parameters provided. The selection of diseases focuses on those with high significance, particularly the most common diseases for life insurance claims in India.
    
Taking things a step further, if the individual is not diagnosed with a disease, machine learning models have been developed to estimate at what age they might contract the disease based on their current medical parameters or lifestyle. However, please note that the accuracy of these models are limited.

#  Format Specifcations /Structure
The APIs expect certain parameters and input data to provide accurate results. Some tests require commonly known parameters such as height, weight, age, and stress degree. However, for more precise outcomes, it is necessary to provide complex parameters such as blood pressure and glucose level for certain models. For better insights into the input data values, please refer to the datasets stored on GitHub (the URL of which can be found at the beginning of each .py file within the DISEASE_PRED folder).

<img width="1112" alt="Screenshot 2023-07-14 at 2 33 44 AM" src="https://github.com/shellyannissa/Compute-Disease-Probability/assets/118563935/3f9fb4d9-08a0-4817-9e5d-2ec630724207">

The API endpoints generally follow the format: http://< ip address >:8080/< route >. For example, http://23.34.113.22:8080/chr_kid.

The IP address can be that of the virtual machine where API is running or "localhost" if it is your local machine. Each route corresponds to a specific model and its associated function. For a better understanding of the function assigned to each route, refer to the file api.py.

The response received from disease predictor models follows the json format: {"acc": {accuracy}, "result": {conclusion}}. Here, "accuracy" indicates the accuracy of the trained machine learning model, and "conclusion" is a binary value indicating the presence of the disease.

<img width="1056" alt="Screenshot 2023-07-14 at 1 01 23 AM" src="https://github.com/shellyannissa/Compute-Disease-Probability/assets/118563935/5045d212-31c1-4eef-ac71-a46d8da26ca8">

The disease predictor API has the format: {"result": {disease}}, where the diagnosed disease is returned as a string.

All the age prediction APIs have the format: {"result": {age}}, where the most probable age of disease occurrence is returned.

<img width="1121" alt="Screenshot 2023-07-14 at 1 03 25 AM" src="https://github.com/shellyannissa/Compute-Disease-Probability/assets/118563935/bee37ced-5552-4156-bc56-c94f4fc9a244">

# Running/Deploying
If you are running the API on your local machine or a virtual machine, follow these steps:

Add the file api.py and all the files within the folder PICKLE(those with extensions .pkl). Ensure that all the files are in the same directory.
Run the following commands (skip installing the libraries or packages you already have):
```
sudo apt install python3
sudo apt install python3-pip
pip install numpy
pip install pandas
pip install scikit-learn
pip install flask
```
And to start the server
```
python3 api.py
```

The API will start running, and incoming requests will be logged in the console.

# Technologies Used
The code in the repository is exclusively written in Python, making use of the libraries 
- numpy
-  scikit-learn
-  pandas

Flask has been employed to set up the APIs for the ml models.

# Disclaimer
Please note that the accuracy of the models may not meet expected standards due to factors such as inconsistent data. Even the largest data model in this repository contains only around 4500 entries as a training set. The age predictor model is highly unpredictable.

Please proceed with caution and consider these limitations while utilizing the APIs provided.
