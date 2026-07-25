"""Configuration shared by every pilot condition."""

from bf_transformer.config import Config


# Training graphs use n <= 8, but the residual layout must already contain the
# node-identity and token slots needed for the registered n <= 16 OOD test.
TRAIN_CONFIG = Config(n_max=16)
