import torch
import numpy as np
import esm


class EsmEmbedding:
    def __init__(self, model="15b"):
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
        model_loader, self.layers = model_map.get(model, model_map["default"])
        self.model, self.alphabet = model_loader()

        # Prepare the batch converter and model
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # Disable dropout for deterministic results
        print("########## Model loaded.")

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Transferred model to GPU")

    def predict(self, data, mode="mean"):
        """
        Generate embeddings for the provided data.

        Parameters:
        - data: A batch of sequences to process.
        - mode (str): Aggregation mode for embeddings. Options are 'mean', 'cls', or others (default: 'mean').

        Returns:
        - dict: A dictionary containing embeddings for specified layers.
        """
        torch.cuda.empty_cache()

        # Convert the input batch to tokens
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)

        # Run the model inference
        with torch.no_grad():
            results = self.model(
                batch_tokens, repr_layers=self.layers, return_contacts=False
            )

        token_representations = {}

        for layer in self.layers:
            token_rep = results["representations"][layer]

            # Process based on the mode
            if mode == "mean":
                layer_embedding = np.mean(
                    token_rep[0, 1 : batch_lens[0] - 1].to(device="cpu").numpy(),
                    axis=0,
                )
            elif mode == "cls":
                layer_embedding = token_rep[0, 0].to(device="cpu").numpy()
            else:
                layer_embedding = np.mean(
                    token_rep[0, 1 : batch_lens[0] - 1].to(device="cpu").numpy(),
                    axis=0,
                )
                layer_cls = token_rep[0, 0].to(device="cpu").numpy()
                token_representations[f"{layer}_cls"] = layer_cls

            token_representations[str(layer)] = layer_embedding

        return token_representations
