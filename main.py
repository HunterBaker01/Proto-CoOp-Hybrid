import random
import torch
import numpy as np
from test import run_experiments

if __name__ == "__main__":
    # Set a seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Run the experiment
    results_df = run_experiments()
