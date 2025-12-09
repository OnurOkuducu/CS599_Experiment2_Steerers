# Define variables
PATHS_TO_LATENTS="/restricted/projectnb/cs599dg/onur/steerers/I2P/2025-11-29 19:18:34.002167/"

BLOCK_NAME="text_encoder.text_model.encoder.layers.9"
D_MODEL="768"

# Run the Python script
python train_ksae.py \
    --paths_to_latents "$PATHS_TO_LATENTS" \
    --block_name "$BLOCK_NAME" \
    --d_in "$D_MODEL" \
    --total_training_tokens "400_000_000" \
    --expansion_factor 4\
    --k 32 
