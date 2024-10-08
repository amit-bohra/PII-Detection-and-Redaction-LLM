import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (AdamW, BertForTokenClassification, 
                          BertTokenizerFast, get_linear_schedule_with_warmup)
from typing import List, Dict



# Define PII types and their corresponding labels
pii_types = {
    "<IP_ADDRESS>": 1,
    "<EMAIL_ADDRESS>": 2,
    "<US_SSN>": 3,
    "<CREDIT_CARD>": 4,
    "<PHONE_NUMBER>": 5,
    "NON_PII": 0
}

# Reverse mapping (integer to label)
int_to_label = {v: k for k, v in pii_types.items()}

# Read data from CSV file
df = pd.read_csv('training_data.csv')


def label_word_by_redacted(actual_word: str, redacted_word: str) -> list[int]:
    """
    Assign a label based on the redacted word.

    Args:
        actual_word (str): The original word from the actual sentence.
        redacted_word (str): The redacted word from the redacted sentence.

    Returns:
        list[int]: A list containing the corresponding label.
    """
    if redacted_word in pii_types:
        return [pii_types[redacted_word]]
    else:
        return [pii_types["NON_PII"]]


# Process the actual_sentence and redacted_sentence columns
cleaned_words_list = []
cleaned_labels_list = []

for index, row in df.iterrows():
    actual_sentence = row['actual_sentence'][:-1]
    redacted_sentence = row['redacted_sentence'][:-1]
    
    # Split sentences into words
    actual_words = actual_sentence.split()
    redacted_words = redacted_sentence.split()
    
    cleaned_words = []
    cleaned_labels = []
    
    # Iterate through both actual and redacted words to assign labels
    for actual_word, redacted_word in zip(actual_words, redacted_words):
        cleaned_word = actual_word.strip(" ")  # Strip punctuation
        cleaned_words.append(cleaned_word)
        
        # Assign labels based on redacted_word
        labels = label_word_by_redacted(cleaned_word, redacted_word)
        
        # Add the labels (single or two depending on date and time format)
        cleaned_labels.extend(labels)
    
    cleaned_words_list.append(cleaned_words)
    cleaned_labels_list.append(cleaned_labels)

# Create a new DataFrame with cleaned_words and cleaned_labels
final = pd.DataFrame({
    'cleaned_words': cleaned_words_list,
    'cleaned_labels': cleaned_labels_list
})

# Display the resulting DataFrame
print(final.head())

# Split the dataset into training, validation, and test sets
train_val_df, test_df = train_test_split(final, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=(1/9), random_state=42)


def insert_cls_sep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert [CLS] and [SEP] tokens into the cleaned words and labels.

    Args:
        df (pd.DataFrame): DataFrame containing cleaned words and labels.

    Returns:
        pd.DataFrame: Updated DataFrame with [CLS] and [SEP] tokens included.
    """
    updated_data = []
    for _, row in df.iterrows():
        words = row['cleaned_words']
        labels = row['cleaned_labels']
        new_words = ['[CLS]']
        new_labels = [-101]

        for i, word in enumerate(words):
            # Append the word and its label
            new_words.append(word.strip('"'))  # Remove quotes and add word
            new_labels.append(labels[i])

            # Check if the word ends with a period or is a period
            if word.endswith('.'):
                # Append [SEP] after a word that ends with a period
                new_words.append('[SEP]')
                new_labels.append(-101)

                # If this word is not the last word, start a new sentence with [CLS]
                if i < len(words) - 1:
                    new_words.append('[CLS]')
                    new_labels.append(-101)

        # Ensure there is a [SEP] at the end of the last sentence if not already there
        if new_words[-1] != '[SEP]':
            new_words.append('[SEP]')
            new_labels.append(-101)

        updated_data.append({'cleaned_words': new_words, 'cleaned_labels': new_labels})

    return pd.DataFrame(updated_data)


# Call the function to insert [CLS] and [SEP] tokens
updated_train_df = insert_cls_sep(train_df)
updated_val_df = insert_cls_sep(val_df)
updated_test_df = insert_cls_sep(test_df)

train = updated_train_df[['cleaned_words', 'cleaned_labels']]
test = updated_test_df[['cleaned_words', 'cleaned_labels']]
val = updated_val_df[['cleaned_words', 'cleaned_labels']]


# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


def sliding_window_tokenization_and_labels(words: list[str], labels: list[int], 
                                           max_len: int = 512, slide_len: int = 256) -> dict:
    """
    Tokenize the input words with sliding window approach and align the labels.

    Args:
        words (list[str]): List of words to tokenize.
        labels (list[int]): List of corresponding labels for the words.
        max_len (int): Maximum length of the tokenized input.
        slide_len (int): Length of the sliding window.

    Returns:
        dict: A dictionary containing tokenized input IDs, attention masks, and labels.
    """
    tokenized_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }
    
    # Process each sequence with sliding windows
    start_index = 0
    while start_index < len(words):
        end_index = start_index + slide_len
        window_words = words[start_index:end_index]
        window_labels = labels[start_index:end_index]

        # Tokenization and padding to max length
        inputs = tokenizer(
            window_words,
            is_split_into_words=True,
            add_special_tokens=False,  # We already have [CLS] and [SEP] in our sequence
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Align labels with the tokenized word IDs, adding -101 for ignored tokens
        word_ids = inputs.word_ids(0)  # Batch index 0 since we're processing one sequence at a time
        window_aligned_labels = [-101 if word_id is None else window_labels[word_id] for word_id in word_ids]

        # Append the tokenized results
        tokenized_inputs['input_ids'].append(inputs['input_ids'].squeeze(0))
        tokenized_inputs['attention_mask'].append(inputs['attention_mask'].squeeze(0))
        tokenized_inputs['labels'].append(torch.tensor(window_aligned_labels, dtype=torch.long))

        # Move start index to the next slide
        start_index += slide_len

    # Stack all the tensors
    tokenized_inputs['input_ids'] = torch.stack(tokenized_inputs['input_ids'])
    tokenized_inputs['attention_mask'] = torch.stack(tokenized_inputs['attention_mask'])
    tokenized_inputs['labels'] = torch.stack(tokenized_inputs['labels'])

    return tokenized_inputs


# Initialize lists to hold the tokenized data for all rows
train_input_ids = []
train_attention_masks = []
train_labels = []

# Tokenize each row in the training DataFrame
for index, row in train.iterrows():
    tokenized_train_data = sliding_window_tokenization_and_labels(
        row['cleaned_words'], row['cleaned_labels']
    )
    train_input_ids.append(tokenized_train_data['input_ids'])
    train_attention_masks.append(tokenized_train_data['attention_mask'])
    train_labels.append(tokenized_train_data['labels'])


# Concatenate the lists of tensors into single tensors
train_input_ids = torch.cat(train_input_ids, dim=0)
train_attention_masks = torch.cat(train_attention_masks, dim=0)
train_labels = torch.cat(train_labels, dim=0)

# Convert tokenized data into a TensorDataset
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

# Define a batch size
batch_size = 16  # You can adjust this according to your requirements

# Create a DataLoader
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)


# Initialize lists to hold the tokenized data for the test set
test_input_ids = []
test_attention_masks = []
test_labels = []

# Tokenize all the data in the test set
for index, row in test.iterrows():
    tokenized_test_data = sliding_window_tokenization_and_labels(
        row['cleaned_words'], row['cleaned_labels']
    )
    test_input_ids.append(tokenized_test_data['input_ids'])
    test_attention_masks.append(tokenized_test_data['attention_mask'])
    test_labels.append(tokenized_test_data['labels'])


# Concatenate the lists of tensors into single tensors
test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)
test_labels = torch.cat(test_labels, dim=0)

# Convert the tokenized test data into a TensorDataset
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# For test DataLoader, we usually don't need to shuffle the data, so we use the SequentialSampler
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Initialize lists to hold the tokenized data for all rows
val_input_ids: List[torch.Tensor] = []
val_attention_masks: List[torch.Tensor] = []
val_labels: List[torch.Tensor] = []

# Tokenize all the data in the validation set
for index, row in val.iterrows():
    """
    Tokenizing the validation data and appending it to the lists.
    
    Args:
        index (int): The index of the row.
        row (pd.Series): The row of the DataFrame containing cleaned words and labels.
    """
    tokenized_val_data: Dict[str, torch.Tensor] = sliding_window_tokenization_and_labels(
        row['cleaned_words'], row['cleaned_labels']
    )
    val_input_ids.append(tokenized_val_data['input_ids'])
    val_attention_masks.append(tokenized_val_data['attention_mask'])
    val_labels.append(tokenized_val_data['labels'])

# Concatenate the lists of tensors into single tensors
val_input_ids = torch.cat(val_input_ids, dim=0)
val_attention_masks = torch.cat(val_attention_masks, dim=0)
val_labels = torch.cat(val_labels, dim=0)

# Convert the tokenized validation data into a TensorDataset
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

# Create a DataLoader for the validation dataset
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)


#########################################################################################


# Check and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device : ', device)

# Define the frequency counts of labels
label_freqs = {
    "NON_PII": 1000000,    # Highest frequency for NON_PII
    "<IP_ADDRESS>": 10000,  # Equal frequency for other PII types
    "<EMAIL_ADDRESS>": 10000,
    "<US_SSN>": 10000,
    "<CREDIT_CARD>": 10000,
    "<PHONE_NUMBER>": 10000
}

# Calculate weights as the inverse of the frequency
weights = 1.0 / torch.tensor(list(label_freqs.values()), dtype=torch.float)

# Normalize the weights so that the most common class (NON_PII) gets a weight of 1
weights = weights / weights[0]  # Assuming 'NON_PII' is the first class

# Move the weights to the device
weights = weights.to(device)

# Loss function using custom weights
loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-101)


# Set training parameters
num_labels: int = len(int_to_label)
batch_size: int = 16
num_epochs: int = 50  # Higher number, since early stopping can halt training
learning_rate: float = 5e-5


# Initialize the model
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps: int = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Early Stopping setup
patience: int = 3  # Number of epochs to wait after last time validation loss improved
best_val_loss: float = float('inf')
best_accuracy: float = 0.0
no_improve_epochs: int = 0


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate the accuracy for a batch of predictions.

    Args:
        logits (torch.Tensor): The predicted logits from the model.
        labels (torch.Tensor): The true labels.

    Returns:
        float: The accuracy of the batch.
    """
    preds = torch.argmax(logits, dim=-1)
    mask = labels != -101  # Exclude the -101 labels from calculation
    corrects = (preds == labels) & mask  # Correct predictions
    accuracy = corrects.sum().item() / mask.sum().item()
    return accuracy


# Set random seeds for reproducibility
seed_val: int = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def get_current_learning_rate(optimizer: AdamW) -> float:
    """
    Get the current learning rate from the optimizer.

    Args:
        optimizer (AdamW): The optimizer used for training.

    Returns:
        float: The current learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


# Begin training loop
initial_model_params = [p.clone() for p in model.parameters()]

# Training loop
for epoch in range(num_epochs):
    print(f"Current Learning Rate: {get_current_learning_rate(optimizer)}")
    total_loss = 0.0
    total_accuracy = 0.0

    model.train()

    for batch in train_dataloader:
        input_ids, attention_masks, labels = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks)

        loss = loss_fn(outputs.logits.view(-1, num_labels), labels.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Calculate batch accuracy from the logits and labels
        batch_acc = batch_accuracy(outputs.logits, labels)
        total_accuracy += batch_acc

    avg_train_loss = total_loss / len(train_dataloader)
    avg_train_accuracy = total_accuracy / len(train_dataloader)
    print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_accuracy}')

    # Validation step
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_masks, labels = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_masks)

            loss = loss_fn(outputs.logits.view(-1, num_labels), labels.view(-1))
            val_loss += loss.item()

            # Calculate batch accuracy from the logits and labels
            batch_acc = batch_accuracy(outputs.logits, labels)
            val_accuracy += batch_acc

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_accuracy = val_accuracy / len(val_dataloader)
    print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}')

    # Check if current epoch's validation loss is the best we've seen so far
    if avg_val_loss < best_val_loss:
        torch.save(model.state_dict(), 'results/best_model_state.bin')
        best_val_loss = avg_val_loss
        best_accuracy = avg_val_accuracy
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            print(f"Best Val Loss: {best_val_loss}, Best Val Accuracy: {best_accuracy}")
            break

# Save the final model state
torch.save(model.state_dict(), 'results/best_model_state.bin')


#################################################################################

# Prepare lists to accumulate true labels and predictions
true_labels_list: List[int] = []
predictions_list: List[int] = []

# Load the best model state
model.load_state_dict(torch.load('results/best_model_state.bin', map_location=device))
model.eval()  # Set the model to evaluation mode

test_loss: float = 0.0
test_accuracy: float = 0.0

# Evaluation loop
with torch.no_grad():  # No gradients needed
    for batch in test_dataloader:
        input_ids, attention_masks, labels = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask=attention_masks)
        loss = loss_fn(outputs.logits.view(-1, num_labels), labels.view(-1))
        test_loss += loss.item()

        # Calculate batch accuracy from the logits and labels
        batch_acc: float = batch_accuracy(outputs.logits, labels)
        test_accuracy += batch_acc

        # Get the true labels and predictions
        preds = torch.argmax(outputs.logits, dim=-1)
        true_labels_list.extend(labels.view(-1).cpu().numpy())
        predictions_list.extend(preds.view(-1).cpu().numpy())

# Compute the average loss and accuracy
avg_test_loss: float = test_loss / len(test_dataloader)
avg_test_accuracy: float = test_accuracy / len(test_dataloader)

# Remove the ignored index (-101) from true labels and predictions
true_labels: List[int] = [label for label in true_labels_list if label != -101]
predictions: List[int] = [pred for pred, true in zip(predictions_list, true_labels_list) if true != -101]

# Print results
print(f'Test Loss: {avg_test_loss}')
print(f'Test Accuracy: {avg_test_accuracy}')

# Generate and print the classification report
report = classification_report(
    true_labels, 
    predictions, 
    labels=list(int_to_label.keys()), 
    target_names=list(int_to_label.values()), 
    zero_division=0
)
print(report)
