# Checkpoints

Model checkpoints are saved in the `checkpoints/` directory.

The repository currently contains:
* `nano_ep55.pt`: A sample checkpoint file. 
*(Note: Ensure you are loading the correct epoch as trained or downloaded).*

## Required External Weights

Photon relies on a frozen MobileCLIP-S1 encoder. The weights must be downloaded to `models/mobileclip_s1.pt`. This is handled automatically if building the project via the provided Dockerfile.
