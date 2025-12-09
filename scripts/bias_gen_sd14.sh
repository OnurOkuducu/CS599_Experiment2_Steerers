CONCEPT_ERASURE="Man"
PROMPT="/restricted/projectnb/cs599dg/onur/steerers/prompts_eval_female.csv"
OUTDIR="bias_result/sd14_exp4_layer9/ethnic_02" 

python bias_gen_sd14.py \
    --concept_erasure "$CONCEPT_ERASURE" \
    --start_iter 0 \
    --prompt "$PROMPT" \
    --strength +0.1 \
    --outdir "$OUTDIR" 