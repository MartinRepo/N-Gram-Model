"""
ANLP Assignment 1
Authors: Chi Xing, Bowen Li
Email: s2682783@ed.ac.uk, s2709967@ed.ac.uk
Date: 2024-10-05
|====================WARM UP=======================|
| Calculate the perplexity of the sequence ##abaab#|
| Perplexity = 2^(- logP / N)                      |
| Result = 3.14                                    |
|==================================================|
"""
from utils import generate_from_LM, compute_perplexity
from models.smoothing_model import smoothing_model_training
from models.backoff_model import backoff_model_training
from models.kn_model import kn_model_training


"""
|====================MAIN SCRIPT=======================|
| Usage: python main.py <training_file> <output_file>  |
|======================================================|
"""
# Test model_training()
smoothing_model_training("assignment1-data/training.en", "assignment1-data/output_model_smoothing.en", "goodTuring")
backoff_model_training("assignment1-data/training.en", "assignment1-data/output_model_backoff.en")

# Test generate_from_LM()
print(generate_from_LM("assignment1-data/model-br.en", 300))
print(generate_from_LM("assignment1-data/output_model_backoff.en", 300))

# Test compute_perplexity()
print(compute_perplexity("assignment1-data/test", "assignment1-data/model-br.en"))
print(compute_perplexity("assignment1-data/test", "assignment1-data/output_model_backoff.en"))
