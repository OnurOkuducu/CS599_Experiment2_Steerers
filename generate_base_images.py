import os, re, csv, torch
from diffusers import DiffusionPipeline

CSV_PATH = "/restricted/projectnb/cs599dg/onur/steerers/prompts_eval.csv"   
NUM_IMAGES_PER_PROMPT = 10              
GUIDANCE = 7.5
STEPS = 50
OUTDIR = "outs_sd14"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,  
)
pipe.to(device)
pipe.set_progress_bar_config(disable=True)

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80]

os.makedirs(OUTDIR, exist_ok=True)

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompt = row["prompt"].strip()
        base_seed = int(row["evaluation_seed"])

        subdir = os.path.join(OUTDIR, slugify(prompt))
        os.makedirs(subdir, exist_ok=True)

        for k in range(NUM_IMAGES_PER_PROMPT):
            seed = base_seed + k
            gen = torch.Generator(device=device).manual_seed(seed)
            out = pipe(
                prompt,
                guidance_scale=GUIDANCE,
                num_inference_steps=STEPS,
                generator=gen,
            )
            img = out.images[0]
            img.save(os.path.join(subdir, f"{k:02d}_seed{seed}.png"))
