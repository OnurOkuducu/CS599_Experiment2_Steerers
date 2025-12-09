CONCEPT_ERASURE="violence, blood"
PROMPT="/restricted/projectnb/cs599dg/onur/steerers/collect_features/datasets/i2p_benchmark.csv"
OUTDIR="violence_result/sd14_exp4_layer9" 

# Run the Python script
python unsafe_gen_sd14.py \
    --concept_erasure "$CONCEPT_ERASURE" \
    --start_iter 0 \
    --prompt "$PROMPT" \
    --strength -0.7 \
    --outdir "$OUTDIR" 