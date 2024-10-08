import pandas as pd
import re
import time
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from typing import Tuple, List

class PIIProcessor:
    """A class for processing PII data and redacting it from sentences."""
    
    def __init__(self, model_path: str):
        """Initialize the PIIProcessor with a pre-trained model and tokenizer."""
        self.pii_types = {
            "<IP_ADDRESS>": 1,
            "<EMAIL_ADDRESS>": 2,
            "<US_SSN>": 3,
            "<CREDIT_CARD>": 4,
            "<PHONE_NUMBER>": 5,
            "NON_PII": 0
        }
        
        # Reverse mapping (integer to label)
        self.int_to_label = {v: k for k, v in self.pii_types.items()}
        
        # Load model and tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        num_labels = len(self.int_to_label)
        self.model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def preprocess_and_redact(self, input_sentence: str) -> Tuple[str, float]:
        """
        Preprocess the input sentence and perform inference using the model.

        Args:
            input_sentence (str): The sentence to be processed.

        Returns:
            Tuple[str, float]: The redacted sentence and inference time.
        """
        start_time = time.time()  # Start timing
        
        inputs = self.tokenizer.encode_plus(
            input_sentence,
            add_special_tokens=True,
            return_tensors='pt',
            padding=False,
            truncation=True
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())
        redacted_sentence = []
        
        for token, pred in zip(tokens, preds):
            if pred in self.int_to_label:
                label = self.int_to_label[pred]
                redacted_sentence.append(label if label != 'NON_PII' else token)
            else:
                redacted_sentence.append(token)

        final_output = self.tokenizer.convert_tokens_to_string(redacted_sentence)
        inference_time = time.time() - start_time  # Calculate inference time
        return final_output, inference_time

    def remove_consecutive_duplicates(self, text: str) -> str:
        """
        Remove consecutive duplicate labels from the redacted text.

        Args:
            text (str): The text with possible consecutive duplicates.

        Returns:
            str: Text with consecutive duplicates removed.
        """
        text = re.sub(r'\[CLS\]|\[SEP\]', '', text)
        return re.sub(r'(<\w+>)(\s+\1)+', r'\1', text)

    def process_csv(self, file_path: str, output_file: str) -> None:
        """
        Process the input CSV file and save the redacted sentences.

        Args:
            file_path (str): Path to the input CSV file.
            output_file (str): Path to save the output redacted CSV file.
        """
        df = pd.read_csv(file_path)
        results = []

        for index, row in df.iterrows():
            actual_sentence = row['actual_sentence']
            redacted_output, inference_time = self.preprocess_and_redact(actual_sentence)
            updated_redacted = self.remove_consecutive_duplicates(redacted_output)
            results.append({
                'original_sentence': actual_sentence,
                'redacted_sentence': updated_redacted,
                'inference_time': inference_time
            })

        # Create a new DataFrame with the results
        results_df = pd.DataFrame(results)
        
        # Save the results to a CSV
        results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    model_path = 'results/best_model_state.bin'  # Path to your model
    processor = PIIProcessor(model_path=model_path)
    csv_file_path = 'validation_data.csv'  # Input CSV file
    output_csv_path = 'redacted_validation.csv'  # Output CSV file
    processor.process_csv(csv_file_path, output_csv_path)
