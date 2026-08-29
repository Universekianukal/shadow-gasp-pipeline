import os, sys, subprocess, json
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
    # Any detected text region at all is a violation — these prompts explicitly
    # forbid text/logos/speech bubbles, so a real still should have none.
    # Confidence > 0.35 filters out pure noise-shaped false positives.
    results = ocr.readtext(np.array(img))
    hits = [r for r in results if r[2] > 0.35]
    return hits

SHOTS = [{"n": 1, "prompt": "A man mid-fall off the rear swim platform of a small catamaran yacht, arm still outstretched toward a hat skimming across the water, his eyes wide with panic, mouth open in a shout, body twisted mid-plunge, ocean spray frozen around him, two other crew members lunging toward the platform edge reaching for him, tropical reef water below, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (teal, rust orange, charcoal), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 2, "prompt": "A small white catamaran yacht drifting alone on a vast empty stretch of ocean near a distant reef coastline, sails limp, no crew visible on deck, morning haze over the water, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (teal, pale grey, charcoal), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 3, "prompt": "A weathered fisherman on the deck of a small trawler, binoculars lowered from his face, eyes wide with alarm, mouth slightly open, staring directly at a drifting yacht in the distance, tropical reef waters behind him, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (teal, rust, charcoal), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 4, "prompt": "Close macro shot of a yacht's diesel engine compartment, throttle lever still pushed forward, engine visibly running, oily metal and cables catching dim light, no people in frame, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (charcoal, rust orange, dull steel), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 5, "prompt": "A rescue officer stepping onto the empty deck of the drifting yacht, eyes wide with disbelief, mouth open mid-gasp, staring down at the untouched deck around him, ropes and slack sails visible nearby, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (teal, rust, charcoal), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 6, "prompt": "Close shot of a yacht's mainsail flapping loose against the mast and rigging, ropes swaying, no people in frame, open ocean visible beyond the boom, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (pale grey, teal, charcoal), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 7, "prompt": "Close macro shot of a small dining table on a yacht's deck, plates of an untouched meal, a knife resting on a cutting board, condensation on a glass, no people in frame, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (rust orange, charcoal, faded yellow), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 8, "prompt": "A rescue officer's face in close-up, eyes wide with unease, mouth slightly parted, staring downward at something off-frame on a table, harsh boat-cabin light on his face, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (teal, charcoal, dull gold), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 9, "prompt": "Close macro shot of an open laptop on a yacht table, screen glowing pale blue-white in dim cabin light, keys slightly worn, no people in frame, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (pale blue, charcoal, rust), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 10, "prompt": "Close macro shot of three life jackets folded neatly and completely dry on a bench seat inside a yacht cabin, straps still buckled, no people in frame, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (rust orange, charcoal, faded white), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 11, "prompt": "An investigator leaning over a table covered in nautical charts, eyes narrowed with intense focus, mouth set in a grim line, one finger pressed against a marked coastline, harsh overhead light, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (charcoal, teal, faded gold), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Queensland, Australia"}, {"n": 12, "prompt": "Close macro shot of a large nautical search map with a grid of red lines drawn across an expanse of open ocean near a reef coastline, pins marking search zones, no people in frame, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (faded parchment, red, charcoal), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Queensland, Australia"}, {"n": 13, "prompt": "A search plane pilot in the cockpit, eyes scanning the horizon with tense focus, mouth pressed tight, endless empty ocean stretching beyond the windshield, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (steel blue, charcoal, pale grey), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 14, "prompt": "A man's arm and shoulder stretched over the edge of a yacht's swim platform, straining toward a floating hat just out of reach on choppy water, his face turned toward it with a look of mild urgent concentration, ocean spray around the hull, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (teal, rust, charcoal), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 15, "prompt": "Three men treading water in open ocean, eyes wide with terror, mouths open mid-shout, arms raised toward their yacht sailing steadily away in the distance under its own power, waves rising around them, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (deep teal, charcoal, rust orange), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}, {"n": 16, "prompt": "A vast empty ocean horizon at sunset near a reef coastline, no boat, no figures, no wreckage, only still water fading into dusk light, dark, moody cel-shaded digital illustration, thick black ink outlines, dramatic cinematic lighting, muted desaturated palette (burnt orange, deep teal, charcoal), noir illustrated aesthetic, single-scene composition, no speech bubbles, no dialogue, no captions, no text or logos, 2007 Great Barrier Reef, Australia"}]
MAX_ATTEMPTS = 3

for s in SHOTS:
    n, p = s["n"], s["prompt"]
    saved = False
    for attempt in range(MAX_ATTEMPTS):
        seed = 3000 + n + attempt * 10000
        try:
            img=pipe(p, num_inference_steps=25, guidance_scale=4.5, height=1280, width=720,
                     generator=torch.Generator("cpu").manual_seed(seed)).images[0]
            m=float(np.asarray(img).mean())
            if m<5:
                print("RETRY", n, "attempt", attempt+1, "black/NaN frame", flush=True)
                torch.cuda.empty_cache()
                continue
            hits = has_text(img)
            if hits:
                texts = [h[1] for h in hits]
                print("RETRY", n, "attempt", attempt+1, "text detected:", texts, flush=True)
                torch.cuda.empty_cache()
                continue
            img.save(f"/kaggle/working/{n:02d}.jpeg", quality=92)
            print("DONE", n, "meanpix", round(m,1), "attempt", attempt+1, flush=True)
            saved = True
            break
        except Exception as e:
            print("FAILED", n, "attempt", attempt+1, repr(e), flush=True)
        torch.cuda.empty_cache()
    if not saved:
        print("GAVE UP", n, "after", MAX_ATTEMPTS, "attempts — saving last generation anyway with a warning", flush=True)
        img.save(f"/kaggle/working/{n:02d}.jpeg", quality=92)
        with open("/kaggle/working/FLAGGED.txt", "a") as f:
            f.write(f"{n:02d}.jpeg needs manual review (artifact after {MAX_ATTEMPTS} attempts)\n")
print("ALL DONE", flush=True)
