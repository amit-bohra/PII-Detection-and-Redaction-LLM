import random
import csv
from faker import Faker
from typing import Dict, List, Callable

class PIIDataGenerator:
    """A class for generating synthetic PII data and sentences."""

    def __init__(self):
        """Initialize the PIIDataGenerator with Faker and PII templates."""
        self.fake = Faker()
        self.pii_templates: Dict[str, Callable[[], str]] = {
            'CREDIT_CARD': lambda: self.fake.credit_card_number(card_type=None),
            'IP_ADDRESS': lambda: self.fake.ipv4(),
            'EMAIL_ADDRESS': lambda: self.fake.email(),
            'PHONE_NUMBER': lambda: self.fake.phone_number(),
            'US_SSN': lambda: self.fake.ssn(),
        }
        self.sentence_templates: Dict[str, List[str]] = {
            'CREDIT_CARD': [
                "{pii} was charged for the recent purchase",
                "You need to verify {pii} as your payment method",
                "The system flagged {pii} for security reasons",
                "A transaction was attempted using {pii}",
                "{pii} seems invalid please check again",
                "Can you please share the last four digits of {pii}"
            ],
            'IP_ADDRESS': [
                "{pii} was used to access the server",
                "We logged an attempt from {pii}",
                "The IP {pii} is flagged for unusual activity",
                "Access was granted to {pii}",
                "The system recorded {pii} as the source of the connection",
                "Please verify if {pii} is your IP address"
            ],
            'EMAIL_ADDRESS': [
                "{pii} received a notification about the update",
                "The system will send an email to {pii}",
                "You can reach the support team via {pii}",
                "We could not deliver the message to {pii}",
                "The email {pii} seems incorrect, please recheck",
                "I registered with {pii} for this service"
            ],
            'PHONE_NUMBER': [
                "{pii} will be used to contact you",
                "I received a message from {pii} earlier",
                "Call us at {pii} for more information",
                "{pii} is unreachable at the moment",
                "The call came from {pii}",
                "Is {pii} still your contact number"
            ],
            'US_SSN': [
                "{pii} was used for the application",
                "Please enter the last four digits of {pii}",
                "The system flagged {pii} for further verification",
                "{pii} is needed to process the form",
                "Someone tried to access the account using {pii}",
                "Is {pii} correct for the records"
            ]
        }

    def generate_sentence(self, pii_type: str, pii_value: str) -> str:
        """
        Generate a sentence with the specified PII entity.

        Args:
            pii_type (str): The type of PII.
            pii_value (str): The value of the PII.

        Returns:
            str: A sentence containing the PII.
        """
        template = random.choice(self.sentence_templates[pii_type])
        position = random.choice(['start', 'middle', 'end'])

        if position == 'start':
            return f"{pii_value} {template.replace('{pii}', '')}".strip()
        elif position == 'middle':
            parts = template.split("{pii}")
            return f"{parts[0].strip()} {pii_value} {parts[1].strip()}"
        else:
            return template.replace("{pii}", f"{pii_value}").strip()

    def create_redacted_sentence(self, sentence: str, pii_value: str, pii_type: str) -> str:
        """
        Create a redacted version of the given sentence.

        Args:
            sentence (str): The original sentence.
            pii_value (str): The PII value to redact.
            pii_type (str): The type of PII.

        Returns:
            str: The sentence with PII redacted.
        """
        return sentence.replace(pii_value, f"<{pii_type}>")

    def generate_dataset(self, num_samples: int, file_name: str) -> None:
        """
        Generate a dataset of sentences with PII and their redacted versions.

        Args:
            num_samples (int): The number of samples to generate.
            file_name (str): The name of the CSV file to write the data to.
        """
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["actual_sentence", "redacted_sentence"])  # Header
            for _ in range(num_samples):
                pii_type = random.choice(list(self.pii_templates.keys()))
                pii_value = self.pii_templates[pii_type]()
                sentence = self.generate_sentence(pii_type, pii_value)
                redacted_sentence = self.create_redacted_sentence(sentence, pii_value, pii_type)
                writer.writerow([sentence, redacted_sentence])

if __name__ == "__main__":
    generator = PIIDataGenerator()
    generator.generate_dataset(500, 'training_data.csv')  # Generate training data
    generator.generate_dataset(100, 'validation_data.csv')  # Generate validation data
