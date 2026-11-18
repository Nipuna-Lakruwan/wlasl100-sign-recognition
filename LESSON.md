# How it works: recognising signs from a webcam with landmarks and a GRU

A lesson for a software engineer moving into AI engineering. Concepts first, then a walk
through the code in the order the data flows.

---

## Part 1: Concepts

### 1.1 The task

**Isolated sign recognition**: a short video of someone signing **one** American Sign
Language (ASL) word goes in; the word comes out. It's a video classification problem with
100 classes. (Translating continuous signing into sentences is a much harder problem; ASL
also has its own grammar, facial expressions and use of space that word-level models ignore.)

### 1.2 The data: WLASL100

**WLASL** (Word-Level ASL, Li et al. 2020) collected 21,000 clips of 2,000 words from ASL
dictionary websites and YouTube, by 119 signers. **WLASL100** is its 100 most frequent words:
2,038 clips, about 20 per word, with an official train/val/test split.

WLASL ships **links**, not videos, and about half the links are dead. We use a Hugging Face
mirror that saved the surviving clips: **1,007 usable clips**, about 7 training examples per
word. That's tiny for deep learning, and it drives every design decision below.

### 1.3 Why landmarks instead of pixels

A raw video model (like I3D, a 3D CNN) must learn from pixels that a hand is a hand, while
also ignoring the background, the lighting, the signer's shirt and the camera angle. With 7
examples per word, it can't.

**MediaPipe** (Google) already solves "where are the hands and body?" with models trained on
huge datasets. It gives us, for every frame:

- **21 hand landmarks** per hand: wrist, and 4 joints for each finger;
- **33 pose landmarks** for the body. We keep 7: nose, shoulders, elbows, wrists.

So a frame becomes 49 points × (x, y) = 98 numbers, plus 2 flags for "left/right hand
visible" = **100 numbers per frame**. A 2-second clip goes from ~3 million pixel values to
5,000 numbers that are all about the signing.

> Software analogy: instead of parsing raw HTML for every request, you call a well-tested
> parser once and work on the DOM. Reusing a strong pretrained component as a fixed feature
> extractor is the most common pattern in applied ML.

### 1.4 Normalising the landmarks

The same sign should give the same numbers whether you sit close or far, left or right in the
frame. So for every frame:

- **origin** = midpoint of the shoulders,
- **unit length** = shoulder width.

A hand at (1, −0.5) now means "one shoulder-width right of centre, half a width up", wherever
the signer is. The tests check this property directly.

**Which hand is which?** MediaPipe labels hands "Left"/"Right", but that label flips with
mirroring (webcams usually mirror). We instead assign each hand to the **nearest pose wrist**,
which doesn't care about mirroring.

### 1.5 A sequence model: the GRU

A sign is **motion**: "drink" and "computer" have similar hand shapes but different paths. We
need a model that reads frames in order.

A **recurrent neural network** keeps a hidden state (a memory vector) and updates it frame by
frame: `h_t = f(h_{t-1}, x_t)`. A **GRU** (Gated Recurrent Unit) adds *gates*: learned switches
that decide how much of the old memory to keep and how much new input to write. That lets it
remember things across many frames without the signal fading away.

Our model:

```
32 frames × 100 features
→ add velocity (each point's movement since the last frame)          32 × 198
→ per-frame MLP (Linear → LayerNorm → GELU → Dropout)                32 × 192
→ 2-layer bidirectional GRU (reads forward AND backward)             32 × 256
→ mean and max over time                                             512
→ Linear                                                             100 word scores
```

**Bidirectional**: one GRU reads the clip forwards, another backwards; their outputs are
joined. For a recorded clip, knowing how the sign ends helps interpret how it started.

Alternatives you'd try next: an LSTM (similar, one more gate), a small **transformer** (attention
over all frames at once), or a graph network treating the skeleton as a graph.

### 1.6 Fighting overfitting with 7 examples per word

With so little data the model memorises training clips quickly (watch the training loss fall
while validation accuracy stalls). Our defences:

- **Data augmentation**: every time a clip is used, a random variant is made: speed and
  start/end changes (temporal crop), size, rotation up to ±15°, shift, small noise, and
  **mirroring** (swap left/right, which is like a left-handed signer).
- **Dropout** (40%), **weight decay**, **label smoothing** (the target is 90% on the right word
  instead of 100%, discouraging overconfidence).
- **Small model** (~0.6 M parameters) and **early stopping** on validation accuracy.

The no-augmentation run in the README shows how much augmentation is worth.

### 1.7 Top-1 vs top-5

**Top-1**: is the highest-scoring word right? **Top-5**: is the right word anywhere in the top 5?
Top-5 matters for products: a demo can show 3 guesses and let the user pick.

### 1.8 A leak hiding in the official split

WLASL's split is by **clip**, not by signer: most test signers also appear in training. The
model can partly recognise *how that person* signs. That's the ECG lesson again. Only 3 of
the 100 surviving test clips are by new signers, far too few to measure anything. So we run a
second experiment: hold out **whole signers** (`train.py --split signer`) from training and
validation, and test on them. Result: 75% top-1 on the official split, 67% on unseen signers.
The second number is the honest estimate for a stranger in front of the webcam.

### 1.9 From clips to a live webcam

Training clips are cut to one sign each. A webcam stream isn't. The demo uses a simple rule:
**hands up = signing**. While at least one hand is visible, frames are collected. When the
hands have been out of view for 8 frames (~0.3 s), the collected segment is trimmed, resampled
to 32 frames, and classified. Exactly the same `trim_to_hands` and `resample` code runs in
training and in the demo, so the model sees the same kind of input in both. Train/serve
consistency is a classic source of production bugs.

---

## Part 2: Code walkthrough

### 2.1 `scripts/download_data.py`

Reads WLASL's JSON (which word, which split, which signer, which URL), keeps the first 100
words, looks up each clip in the Voxel51/WLASL Hugging Face mirror (pinned to one commit), and
downloads only those. Retries dropped connections; treats 404 as "gone" and records
which clips are available. Also downloads MediaPipe's two `.task` model files.

### 2.2 `src/sign/landmarks.py`

- `normalise()`: pose and hands (pixel coordinates) → the 100-number vector (1.4).
- `Extractor`: wraps MediaPipe's `HandLandmarker` and `PoseLandmarker` in **VIDEO** mode,
  which tracks landmarks from frame to frame (faster and steadier than detecting every frame
  from scratch). Timestamps must increase.
- `video_features()`: a whole file → `(n_frames, 100)`.

### 2.3 `scripts/extract_landmarks.py`

Runs MediaPipe over all clips once, in 3 processes, and caches the result (13 MB). This is
the slow step (~15 min). Training then never touches video, so experiments take minutes.

### 2.4 `src/sign/model.py`

- `resample()`: any clip length → 32 evenly spaced frames.
- `trim_to_hands()`: drops the "standing still" frames before and after the sign.
- `add_velocity()`: `torch.diff` along time: how each point moved since the previous frame.
- `SignGRU`: the model in 1.5. `h.mean(1)` and `h.amax(1)` summarise the 32 time steps two ways:
  the average, and the strongest moment.

### 2.5 `scripts/train.py`

- `augment()`: the random variants in 1.6. Note the mirror: `_FLIP` swaps left/right pose
  points and the two hand slots, then x is negated.
- The loop: batches of 32 clips, `AdamW`, a **one-cycle** learning-rate schedule (warm up, then
  decay), gradient clipping (stops a single bad batch from blowing up the weights).
- Early stopping keeps the epoch with the best validation top-1.

### 2.6 `scripts/evaluate.py`

Averages 3 seeds' probabilities (the headline, chosen before scoring test), computes top-1/5/10,
splits by seen vs new signers, and lists the most-confused word pairs.

### 2.7 `scripts/live_demo.py`

- `Recogniser.step()`: the hands-up/hands-down state machine from 1.9.
- The frame given to MediaPipe is **not** mirrored (like the training videos); only the
  displayed image is mirrored, so it feels like a mirror to you.
- `--video file.mp4 --headless` runs the same pipeline on a file and prints predictions. That's
  how it was tested without a camera.

---

## Part 3: Things to try next

1. **More data**: WLASL300 or WLASL2000 clips from the same mirror (many more signers),
   training on 100 words but learning from more people.
2. **Hand shape detail**: angles between finger joints (rotation-invariant) as extra features.
3. **A small transformer** over frames, compared fairly with the GRU (same data, same seeds).
4. **A "no sign" class** so the live demo can say "I don't know" instead of always guessing.

## Interview questions you should now be able to answer

- Why use MediaPipe landmarks instead of training on raw video here?
- What does normalising by shoulder width buy you?
- What do a GRU's gates do?
- Why is WLASL's official split optimistic, and how would you measure that?
- What's train/serve skew, and how does this project avoid it?
