import re, numpy as np, soundfile as sf, kokoro_onnx, os
HOME = os.path.expanduser('~')
base = HOME + '/.cache/hyperframes/tts'
model = kokoro_onnx.Kokoro(base + '/models/kokoro-v1.0.onnx', base + '/voices/voices-v1.0.bin')
text = open('tc_narration.txt', encoding='utf-8').read()
sents = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
chunks = []; cur = ''
for s in sents:
    s = s.strip()
    if not s: continue
    if len(cur) + len(s) + 1 <= 350: cur = (cur + ' ' + s).strip()
    else:
        if cur: chunks.append(cur)
        cur = s
if cur: chunks.append(cur)
print('chunks:', len(chunks))
sr = 24000; gap = np.zeros(int(0.35 * sr), dtype=np.float32); parts = []
for i, c in enumerate(chunks):
    samples, sr = model.create(c, voice='bm_george', speed=0.92)
    parts.append(np.asarray(samples, dtype=np.float32)); parts.append(gap)
    print(f'  {i+1}/{len(chunks)} ok ({len(samples)/sr:.1f}s)')
audio = np.concatenate(parts)
sf.write('tc_narration.wav', audio, sr)
print('TOTAL %.1f sec  -> tc_narration.wav' % (len(audio) / sr))
