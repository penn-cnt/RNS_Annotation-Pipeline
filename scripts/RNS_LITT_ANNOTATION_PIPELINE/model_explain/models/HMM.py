import numpy as np
import torch

class HMM:
    def __init__(self,predictions):

        self.predictions = predictions
        self.transition_probability = self.calculate_transition_matrix(self.predictions)
        self.init_probability = self.calculate_initial_probabilities(self.predictions)

    def inspect_array(self, name, arr):
        print(f"Details for {name}:")
        print(f"Shape: {arr.shape}")
        print(f"Data Type: {arr.dtype}")
        print(f"Number of Dimensions: {arr.ndim}")
        print(f"First few elements: {arr[:2]}")  # Adjust the number as needed
        print("--------------------------------------------------")


    def calculate_transition_matrix(self, predictions):
        # Initialize a 2x2 matrix to store transition counts
        transition_counts = np.zeros((2, 2), dtype=int)

        # Iterate over each sequence in predictions
        for sequence in predictions:
            for i in range(len(sequence) - 1):
                # Increment the count for the observed transition
                transition_counts[sequence[i], sequence[i + 1]] += 1

        # Normalize the counts to get probabilities
        transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)

        return transition_probs


    def calculate_initial_probabilities(self, predictions):
        # Count the number of sequences that start with 0 and 1
        initial_counts = np.zeros(2, dtype=int)

        for sequence in predictions:
            initial_counts[sequence[0]] += 1

        # Normalize the counts to get probabilities
        initial_probs = initial_counts / len(predictions)

        return initial_probs


    # Forward algorithm
    def forward(self, sequence, emission_probs):
        alpha = np.zeros((len(sequence), 2))
        alpha[0] = self.init_probability * emission_probs[0]

        for t in range(1, len(sequence)):
            for j in range(2):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_probability[:, j]) * emission_probs[t, j]

        return alpha

    # Backward algorithm
    def backward(self, sequence, emission_probs):
        beta = np.zeros((len(sequence), 2))
        beta[-1] = 1

        for t in range(len(sequence)-2, -1, -1):
            for j in range(2):
                beta[t, j] = np.sum(self.transition_probability[j, :] * emission_probs[t+1] * beta[t+1])

        return beta

    # Predict labels using Forward-Backward algorithm
    def predict_labels(self, sequence, emission_probs):
        alpha = self.forward(sequence, emission_probs)
        beta = self.backward(sequence, emission_probs)
        gamma = alpha * beta / np.sum(alpha[-1])
        return np.argmax(gamma, axis=1)