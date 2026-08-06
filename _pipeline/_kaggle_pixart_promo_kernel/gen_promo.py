import os, sys, subprocess
def pip(*a): subprocess.run([sys.executable,"-m","pip","install","-q",*a], check=False)
pip("torch==2.4.1","torchvision==0.19.1","--index-url","https://download.pytorch.org/whl/cu121")
pip("diffusers==0.32.2","transformers==4.46.3","accelerate","sentencepiece","protobuf")
pip("easyocr")

import torch, numpy as np
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)

from diffusers import PixArtSigmaPipeline
pipe = PixArtSigmaPipeline.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
print("PIPE READY", flush=True)

import easyocr
ocr = easyocr.Reader(["en"], gpu=True)
print("OCR READY", flush=True)

def has_text(img):
    results = ocr.readtext(np.array(img))
    return [r for r in results if r[2] > 0.35]

PROMPT = """Noir true-crime aesthetic, cinematic photo-illustration, gritty halftone texture, 1970s muted color palette (mustard yellow, rust orange, faded teal, deep shadow). A weathered wooden desk shot from a high angle, empty rectangular space left clear in the center-left third of the frame for a book to be placed later. Scattered around the edges: a stack of worn cash bundles bound with paper straps, an open case file folder with blank pages, a lit cigarette resting in a glass ashtray with smoke curling up, a pair of dark sunglasses, a coiled length of parachute cord. Rain streaks down a window in the background, city runway lights blurred and bokeh beyond the glass. Dramatic low side lighting, deep shadows, high contrast. No text, no logos, no watermark, no signage, no readable writing anywhere, no book or rectangular object in the empty center space."""
MAX_ATTEMPTS = 4
for attempt in range(MAX_ATTEMPTS):
    seed = 9000 + attempt * 1000
    img = pipe(PROMPT, num_inference_steps=28, guidance_scale=4.5, height=896, width=1152,
               generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    m = float(np.asarray(img).mean())
    if m < 5:
        print("RETRY attempt", attempt+1, "black/NaN frame", flush=True)
        torch.cuda.empty_cache()
        continue
    hits = has_text(img)
    if hits:
        print("RETRY attempt", attempt+1, "text detected:", [h[1] for h in hits], flush=True)
        torch.cuda.empty_cache()
        continue
    img.save("/kaggle/working/norjak_promo_scene.jpeg", quality=94)
    print("DONE attempt", attempt+1, "meanpix", round(m,1), flush=True)
    break
else:
    print("FAILED all attempts", flush=True)
