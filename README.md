# PII Detection and Redaction with LLM

This project implements a system for detecting and redacting Personally Identifiable Information (PII) using a fine-tuned **BERT** model. The project consists of data preparation, model training, and inference.

## Environment Setup

1. **Create a Python environment:**  
   Create a new environment named **redaction-env** using Python.

2. **Install required packages:**  
   Install the required libraries by running the command:

   pip install -r requirements.txt

## Data Preparation

1. **Generate the dataset:**  
   Run the **create_dataset.py** script to create two CSV files: **training_data.csv** and **validation_data.csv**.

## Model Training

1. **Fine-tune the BERT model:**  
   Execute the **finetune_model.py** script, which preprocesses the data, creates data loaders for **PyTorch**, and fine-tunes the **BERT** model. The trained model will be saved in the **results** folder.

2. **Training Logs:**  
   The training process logs will be stored in **training_log.txt**.

## Model Validation and Inference

1. **Validate the model:**  
   Use the **validate_model.py** script to read the trained model and perform redaction on the validation data. The results will be stored in **redacted_validation.csv**, including the original text, redacted text, and inference time.

## Model Requirements

To run the inference successfully, ensure the following file is present:

- **results/best_model_state.bin** - This file should be located at the root level inside the **results** folder.  
  This file can be downloaded using the following [Google Drive link](https://drive.google.com/file/d/1u1Y4gFwciDA0Lwk27tuVgjRGDu4xWZxd/view?usp=sharing).

## PII Types

The system is designed to recognize the following types of PII:

- **CREDIT_CARD:** Randomly generated credit card numbers.
- **IP_ADDRESS:** Valid IPv4 addresses.
- **EMAIL_ADDRESS:** Fake email addresses.
- **PHONE_NUMBER:** Fake phone numbers.
- **US_SSN:** Social Security Numbers.

Each type of PII is generated using the **fake** library to ensure the variety and relevance of the data.

## Future Prospects

1. **Redact more information:**  
   Expand the system to recognize and redact additional types of PII, such as driver's license numbers, passport numbers, etc.

2. **Collect real-time data:**  
   Implement a pipeline to collect and process real-time data for PII detection and redaction.

3. **Fine-tune another model:**  
   Experiment with different models or architectures to improve detection and redaction performance.

4. **Mixed approach:**  
   Combine the current model-based approach with regex for better coverage and accuracy in PII detection and redaction.

5. **User Interface:**  
   Develop a web-based or desktop application for easier access and usability of the PII detection and redaction functionalities.

6. **Performance Optimization:**  
   Improve the model's inference speed and accuracy through optimization techniques.

7. **Integration with Existing Systems:**  
   Explore integrating this solution into existing data processing systems for seamless operation.
