# Dataset Reference — PhC-C2DH-U373

This document records the exact structure of the dataset as verified by direct file inspection. Every number here comes from reading the files on disk; none are assumed.

---

## 1. Directory Structure

```
dataset/train/
├── 01/                     115 TIFs  t000.tif … t114.tif        (images)
├── 01_GT/
│   ├── SEG/                 15 TIFs  man_seg{frame}.tif         (gold outlines)
│   └── TRA/                116 files 115 man_track{frame}.tif + man_track.txt
├── 01_ST/
│   └── SEG/                115 TIFs  man_seg{frame}.tif         (silver outlines)
├── 01_ERR_SEG/             115 TIFs  mask{frame}.tif            (relabelled ST, unused)
├── 02/                     115 TIFs
├── 02_GT/
│   ├── SEG/                 19 TIFs
│   └── TRA/                116 files
├── 02_ST/SEG/              115 TIFs
└── 02_ERR_SEG/             115 TIFs
```

- **Sequences:** `01` and `02`.
- **Frames per sequence:** exactly 115, numbered `t000`–`t114`, no gaps.
- **Image format:** 520 × 696, 8-bit grayscale TIFF.
- **Consecutive pairs available:** 114 per sequence → **228 total**.
- **Auxiliary file:** `man_track.txt` in each `TRA/` folder, storing cell lineage as `cell_id start_frame end_frame parent_id`.

---

## 2. Annotation Types — Exact Properties

### 2.1 GT/SEG — Gold-standard sparse full segmentations

| Property | Value |
|---|---|
| dtype | uint16 |
| Pixel value | Cell ID (0 = background) |
| Frames annotated, seq01 | `[1, 5, 6, 7, 21, 49, 59, 67, 72, 75, 92, 96, 100, 102, 112]` — 15 frames |
| Frames annotated, seq02 | `[10, 17, 19, 20, 22, 26, 35, 36, 41, 49, 51, 52, 59, 74, 77, 85, 92, 106, 112]` — 19 frames |
| Image coverage | 5.5–10.0% per frame (seq01), 2.7–6.8% (seq02) |
| Per-cell area | 1000–13 000 px, mean ≈ 4000 |
| Object count per frame | 3–7 |

**ID stability across time**
- **Seq01:** Labels approximately correspond to tracking IDs. Cells 1–6 use consistent labels throughout the 15 annotated frames. The cell that TRA/ST call "8" (appearing from frame 054) is labelled "7" in GT/SEG from frame 059 onward. This means **GT/SEG seq01 label 7 does NOT match TRA/ST label 7**.
- **Seq02:** Labels are **per-frame spatial indices**, not tracking identities. Verified on frame 010: SEG labels 1–5 correspond respectively to TRA/ST labels 1, 3, 4, 11, 14 by spatial position. Across different frames, the same SEG integer may refer to a different physical cell as the set of cells changes.

**Consequence**: Do not assume `GT/SEG_label == tracking_ID`. Always spatially match GT/SEG labels to ST/TRA labels via pixel overlap when both are needed.

### 2.2 GT/TRA — Centroid markers, full temporal coverage

| Property | Value |
|---|---|
| dtype | uint16 |
| Pixel value | Persistent tracking ID (0 = background) |
| Frames present | All 115 per sequence |
| Per-object bounding box | ~13 × 13 px (seq01), smaller in seq02 |
| Per-object area | 25–250 px (seq01), 9–193 px (seq02) |
| Image coverage | 0.25–0.51% per frame (seq01), 0.05–0.17% (seq02) |

**These are markers, not segmentations.** Each "object" is a small blob placed near a cell's centroid. The TRA label for cell *k* at frame *t* always falls inside the ST region for cell *k* at frame *t* (verified on frame 000, seq01, all cells).

**ID stability:** IDs are stable across a cell's lifetime. Cells are born with a fresh ID, live continuously with no gaps, and die at a specific frame. IDs are never reused.

**Full lineage, seq01** (8 unique IDs):
```
ID 1: 000-114 (full)          ID 5: 000-114 (full)
ID 2: 000-114 (full)          ID 6: 000-114 (full)
ID 3: 000-114 (full)          ID 7: 009-022 (14 frames)
ID 4: 000-114 (full)          ID 8: 054-114 (61 frames)
```

**Full lineage, seq02** (12 unique IDs):
```
ID  1: 000-079           ID  7: 024-030           ID 14: 000-072
ID  2: 089-114           ID  9: 033-114           ID 18: 030-105
ID  3: 000-114           ID 11: 000-058
ID  4: 000-114           ID 12: 068-114
ID  5: 000-003           ID  6: 014-021
```

### 2.3 ST/SEG — Silver Truth, full outlines + full coverage + stable IDs

| Property | Value |
|---|---|
| dtype | uint16 |
| Pixel value | Tracking ID matching TRA (0 = background) |
| Frames present | All 115 per sequence |
| Per-cell area | 1000–13 000 px, mean ≈ 4000 (seq01); ≈ 3000 (seq02) |
| Image coverage | 5.5–10.4% per frame (seq01), 3.4–6.5% (seq02) |

**Relationship to TRA.** For every one of the 230 frames, the set of non-zero labels in ST equals the set of non-zero labels in TRA. ST is effectively "take TRA's markers and extend each one to the full cell body it refers to".

**Relationship to GT/SEG** (where both exist):

| Sequence | Mean binary IoU | Range | Mean per-cell Dice |
|---|---|---|---|
| 01 | **0.956** | 0.94–0.96 | 0.95–0.99 |
| 02 | **0.878** | 0.59–0.95 | 0.76–0.99 |

Poor seq02 agreement concentrates on frames **026, 051, 085, 106** — cases where cells touch or merge and the automatic curator disagrees with the human annotator about boundaries.

**Temporal area variation.** ST cell areas change smoothly and significantly over time (e.g. seq01 cell 6: 3985 → 13319 px across its lifetime). This reflects real biological deformation, not annotation noise.

### 2.4 ERR_SEG — Unused

`NN_ERR_SEG/mask{frame}.tif` contains the same per-frame cell coverage and label count as ST, but label identities are permuted so that corresponding labels have IoU = 0. This is an intentional ID-shuffled version. **It is not used in this project.**

---

## 3. ID Behaviour Across Time — Cross-Annotation Summary

| Question | GT/SEG | GT/TRA | ST/SEG |
|---|---|---|---|
| Same ID = same cell across frames? | seq01 yes-ish, seq02 NO | Yes | Yes |
| IDs reused after cell death? | — | No | No |
| Birth events in annotation? | — | Yes (new ID appears) | Yes (new ID appears) |
| Death events in annotation? | — | Yes (ID stops) | Yes (ID stops) |
| Labels shared with other annotation? | Only partially with TRA/ST | Identical to ST | Identical to TRA |

**The critical fact:** TRA and ST share exactly the same label scheme on every frame. They are the only two annotation layers that can be combined without spatial re-matching.

---

## 4. Valid vs Invalid Assumptions

### Valid assumptions (safe to rely on)
- Every frame in both sequences has both TRA and ST annotations.
- TRA and ST use the same tracking IDs.
- A cell ID's lifetime is a single contiguous range of frames — no gaps, no reuse.
- Within its lifetime, a given ID refers to one and only one physical cell.
- ST cell boundaries are approximately correct (mean IoU > 0.87 with GT where checked).
- Image dimensions are uniform: 520 × 696.
- All 230 images exist with no missing frames.

### Invalid assumptions (do not make)
- "GT/SEG label k = TRA label k." False in seq01 from frame 059 and globally in seq02.
- "The same set of cells is present in every frame." False: 16 of 228 pairs involve a birth or death.
- "ST is pixel-perfect." False: seq02 has a few frames where ST disagrees with GT.
- "Labels 1..N are all the cells." Seq02 uses non-contiguous IDs (1, 3, 4, 9, 11, 12, 14, 18). Iterate over `np.unique(mask)[1:]`, never over `range(1, N+1)`.
- "Centroids and cell bodies are the same thing." TRA provides centroids only.
- "A large `NN_ERR_SEG/` file is a better version of ST." It is a relabelled (shuffled) version and unusable.
- "The 5 pairs usable under GT/SEG are representative of the dataset." They are 2% of the pairs and 4 of 5 come from seq02.
