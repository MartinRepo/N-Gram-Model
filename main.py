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
from models.general_model import model_training
from models.add_alpha import model_training_with_add_alpha
from models.interpolation import model_training_with_interpolation

"""
|====================MAIN SCRIPT=======================|
| Usage: python main.py <training_file> <output_file>  |
|======================================================|
"""
# For searching best alpha
# model_training_with_add_alpha("assignment1-data/training.en", "assignment1-data/model-add-alpha.en")

# For searching best lambdas
# model_training_with_interpolation("assignment1-data/training.en", "assignment1-data/model-interpolation.en")

# Test general model_training(), with different smoothing methods
model_training("assignment1-data/training.en", "assignment1-data/output_model_smoothing.en", "add-alpha", 0.151)

# Draw probability distribution of trigrams with history "ng"
# plot_distribution("assignment1-data/output_model_smoothing.en", "ng")

# Test generate_from_LM()
print(generate_from_LM("assignment1-data/model-br.en", 300))
print(generate_from_LM("assignment1-data/output_model_smoothing.en", 300))

# Test compute_perplexity()
print(compute_perplexity("assignment1-data/test", "assignment1-data/model-br.en"))
print(compute_perplexity("assignment1-data/test", "assignment1-data/output_model_smoothing.en"))
