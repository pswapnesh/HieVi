import torch
import numpy as np
import esm


class EsmEmbedding:
    def __init__(self, model="3b"):
        """
        Initialize the ESM embedding model based on the specified version.

        Parameters:
        - model (str): The model version to load. Options are '650m', '15b', '3b', or others (default: '15b').
        """
        # Load model and define layers based on the version
        model_map = {
            "650m": (esm.pretrained.esm2_t33_650M_UR50D, [33]),
            "15b": (esm.pretrained.esm2_t48_15B_UR50D, [47]),
            "3b": (esm.pretrained.esm2_t36_3B_UR50D, [35]),
            "default": (esm.pretrained.esm2_t6_8M_UR50D, [5]),
        }
        self.device = "cuda"
        model_loader, self.layers = model_map.get(model, model_map["default"])
        self.model, self.alphabet = model_loader()

        # Prepare the batch converter and model
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # Disable dropout for deterministic results
        print("########## Model loaded.")
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.cuda()
            print("Transferred model to GPU")
        else:
            self.device = "cpu"
            print("GPU not available. Running on CPU.")

    def predict(self, data):
        """
        Generate embeddings for the provided data.

        Parameters:
        - data: A batch of sequences to process.

        Returns:
        - np.ndarray: A numpy array containing the embedding vector.
        """
        

        # Convert the input sequence to tokens
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(device=self.device, non_blocking=True)

        # Run the model inference
        with torch.no_grad():
            results = self.model(
                batch_tokens, repr_layers=self.layers, return_contacts=False
            )

        # Extract embeddings for the last layer (no dictionary return, just the vector)
        layer = self.layers[-1]  # Using only the last layer
        token_rep = results["representations"][layer]

        return token_rep[0, 1 : -1].mean(axis=0),token_rep[0, 0]  # This will return a tensor if kept on GPU
