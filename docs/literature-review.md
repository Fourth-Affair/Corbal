# Floor-plan recognition since CubiCasa5K (2019)

What has happened since the CubiCasa5K paper, what applies to our pipeline, and what to build next.
Written November 2026 for the Junction 2026 Stoa team.

**Summary.** Since 2019 the field has moved from per-pixel segmentation plus hand-written post-processing
to (a) object detectors that find each symbol separately, (b) transformers that output room polygons or
wall graphs directly, and (c) point- and graph-based models that read CAD vector drawings without
rasterising them. Two results matter most for us:

- An off-the-shelf RT-DETR detector trained on the official CubiCasa5K split reaches **~89 AP50 on doors and
  windows**. Our pipeline is weakest exactly there (doors 51%, windows 38% exact count), because the
  CubiCasa segmentation merges neighbouring symbols.
- Frontier vision-language models are **still poor at counting line-art symbols** (≤ ~40% exact on
  doors/windows in AECV-Bench, 2026), but good on rooms that have text labels. This supports our design:
  deterministic evidence and small dedicated models first, VLMs only on crops.

So the next step is a small instance detector (YOLO11 / RT-DETR) for doors, windows and fixtures. Its
counts go into our existing fusion models, and its boxes become citations.

---

## 1. The baseline we use: CubiCasa5K (2019)

- **Dataset.** 5,000 Finnish floor plans (three styles: `colorful`, `high_quality`,
  `high_quality_architectural`) with SVG annotations of more than 80 object classes. The official split is
  4,200 train / 400 val / 400 test. Our challenge PNGs are byte-identical to the originals, and the
  challenge questions were generated from the SVGs (our SVG oracle scores 100%).
- **Model.** A multi-task hourglass network (`hg_furukawa_original`, vendored in
  `backend/app/cubicasa/vendor/floortrans`) with 44 output channels: 21 junction heatmaps (wall corners,
  opening endpoints, icon corners), 12 room classes and 11 icon classes. The tasks are balanced with
  uncertainty-weighted losses (Kendall et al.).
- **Post-processing.** Heuristic: junctions → walls → polygons, inherited from *Raster-to-Vector*
  (Liu et al., ICCV 2017), which used integer programming over junctions.
- **Contemporary work.** *Deep Floor Plan Recognition* (DFPR, Zeng et al., ICCV 2019): a VGG encoder with
  room-boundary-guided attention. Many later papers compare against DFPR and CubiCasa5K.
- **Known weakness, which we see too.** Icons are a per-pixel *class*, not instances. Touching windows
  along a wall, or a door next to a window, become one blob. Counting connected components then
  undercounts, and our hand-built "split merged openings" features only partly fix that.

## 2. Raster floor-plan parsing since 2019

### 2.1 Structured / vector outputs (rooms and walls as polygons or graphs)
| Work | Idea | Relevance |
|---|---|---|
| RoomFormer (Yue et al., CVPR 2023) | Single-stage transformer with two levels of queries (rooms → corners) outputs room polygons directly; can be extended to doors/windows | Mostly evaluated on point-cloud density maps (Structured3D, SceneCAD) |
| PolyRoom (ECCV 2024) | Room-aware queries initialised from instance segmentation; beats RoomFormer on Structured3D | Same domain as RoomFormer |
| Raster-to-Graph (Hu et al., Eurographics 2024) | Autoregressive transformer predicts the wall-junction graph in traversal order; releases ~10k real residential plans | Clean room topology; heavier than we need for counting |
| Raster2Seq | Floor-plan elements (rooms, windows, doors) as labelled polygon sequences | Unifies rooms and openings |
| Sparse point-set vectorisation (Automation in Construction, 2025) | Each element as four points; reports state of the art on CubiCasa5K; introduces the CFP dataset | Lightweight vector output |

**For us:** these give cleaner room polygons (better citations, room adjacency), but room counting is
already 77–95% and limited mostly by text labels. They are not the next bottleneck.

### 2.2 Better segmentation
- **CAB1/CAB2 multi-unit recognition** (arXiv 2408.01526, 2024): multi-scale segmentation beats DFPR and
  CubiCasa5K on a cleaned CubiCasa subset. It also reports that **CubiCasa5K annotation quality is
  noticeably lower than R3D's**, which caps how high any exact-count score can go.
- **MitUNet** (arXiv 2512.02413, 2025): SegFormer (Mix-Transformer) encoder + U-Net decoder with a
  Tversky loss; pre-trained on CubiCasa5K, fine-tuned on 500 regional plans.
- **CLIP-guided semantic recognition** (ASCE J. Comput. Civ. Eng.): F1 on CubiCasa5K rises from 64.2 to
  84.0 when CLIP features guide room semantics.

### 2.3 Object detection of symbols (the most relevant strand)
- **RT-DETR-L on the official CubiCasa5K split**
  ([OldDeLorean/rtdetr-floorplan-detector](https://github.com/OldDeLorean/rtdetr-floorplan-detector),
  linked to arXiv 2606.10953). Results on the 400-image test set: door AP50 89.8 / AP 57.5, window
  AP50 89.3 / AP 62.2. Training took 60 epochs at 1024 px, about 2.3 h on 2×A100, and inference is about
  2 s per image on CPU. Weights are public; the repository has no licence file.
- **DoorDet** (arXiv 2508.07714, 2025): about 5k CubiCasa-derived plans. A detector finds doors as one
  class, then an LLM classifies each door's type. This is the same "small model finds, LLM labels" split
  we use.
- **Community YOLO models** on Roboflow (YOLOv8s ~60 mAP50, YOLOv12s ~81 mAP50 on door/window/wall).
  These use different class sets and splits, so they only show that small YOLOs work on this data.

## 3. CAD vector drawings: symbol spotting (for the challenge's PDF/DXF files)
Many of the challenge files are vector PDFs or DXFs. There, symbols are primitives (lines, arcs), not
pixels.
- **FloorPlanCAD** (Fan et al., ICCV 2021): more than 10k CAD floor plans; defines *panoptic symbol
  spotting* (countable symbols plus "stuff" such as walls).
- **CADTransformer**, **GAT-CADNet** (CVPR 2022): transformer / graph-attention models over primitives.
- **SymPoint** (ICLR 2024): primitives as point sets plus a point transformer and a Mask2Former-style head;
  83.3 PQ on FloorPlanCAD. **SymPoint-V2** (2024) adds CAD layer information and reaches 90.1 PQ.
  **CADSpotting** (2024) targets large drawings.
- **ArchCAD-400K** (2025): large CAD dataset and a new baseline.

**For us:** when we parse vector PDFs/DXFs we should keep the primitives (with layer names) and spot
symbols on them directly. This is more precise than rasterising, and citations map straight to entities.

## 4. Foundation models
- **Open-world / exemplar counting.** *CountGD* (Amini-Naieni, Han, Zisserman, NeurIPS 2024) builds on
  GroundingDINO; the target is given as text, exemplar boxes, or both. Follow-ups: CountGD++, CountEx,
  RECOUNT. Useful as a zero-shot fallback for symbol types we have no training data for: the user, or a
  first pass, marks one example and the model counts the rest.
- **VLMs on drawings.** *AECV-Bench* (arXiv 2601.04819, 2026) tests 120 floor plans, partly from
  CubiCasa5K. Bedroom and toilet counts reach 0.74–0.91 exact match, but doors and windows only
  0.09–0.39, with 20–50% mean absolute percentage error. Typical failures: door swings not told apart,
  windows confused with façade lines.

## 5. Takeaways for our pipeline
1. **Counting symbols needs instances, not pixel classes.** A detector separates touching symbols, which
   the CubiCasa icon segmentation cannot.
2. **Small models are enough.** A ~9M-parameter YOLO11s or a ~32M-parameter RT-DETR-L trains in a few
   GPU-hours and runs on CPU in seconds. That fits the "won't be obsoleted by next year's frontier model"
   judging criterion: AECV-Bench shows frontier VLMs are still far behind on exactly these symbols.
3. **Keep the fusion layer.** Detector counts become *more features* for our per-template models, which
   CV picks only where they help. Detector boxes give tighter citations than segmentation blobs.
4. **Text labels drive room counts.** Bedroom (77%) and WC-room (85%) errors are mostly OCR misses, so a
   stronger OCR model is the next lever there.
5. **Watch label noise.** CubiCasa SVGs have known annotation errors. Part of the remaining gap is
   unreachable, and exact match punishes it.

## 6. Roadmap (ranked by expected gain / effort)
1. **Symbol detector** (door, window, toilet, sink, double_sink, shower, bathtub, sauna_stove):
   YOLO11s first, RT-DETR-L as a comparison. Train on a rented Verda GPU, export ONNX, run on CPU at runtime. Features
   go into the fusion models and boxes into citations. Expected to help doors, windows, sinks and
   showers+baths. **(implemented: `app/cubicasa/symdet.py`, `scripts/verda_train.py`)**
2. **Better OCR for room labels**: PP-OCRv5 or Florence-2-base as a second reader → bedrooms, WC rooms,
   bathrooms.
3. **Room segmentation refresh** (SegFormer/MitUNet-style fine-tune): only if 1–2 plateau.
4. **Beyond cubicasa:** vector PDF/DXF parsing that keeps primitives and layers; SymPoint-style spotting;
   CountGD or VLM-on-crops as a zero-shot fallback for unseen symbol types.

**Licences.** CubiCasa5K data and pretrained weights are CC BY-NC (non-commercial). Ultralytics is
AGPL-3.0. Both are fine for a hackathon entry; revisit before any commercial use.

## Sources
- CubiCasa5K — [paper](https://www.researchgate.net/publication/333230440_CubiCasa5K_A_Dataset_and_an_Improved_Multi-task_Model_for_Floorplan_Image_Analysis)
- RoomFormer — [code](https://github.com/ywyue/RoomFormer), [paper](https://arxiv.org/abs/2211.15658)
- PolyRoom — [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06708.pdf)
- Raster-to-Graph — [code](https://github.com/SizheHu/Raster-to-Graph), [paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.15007)
- Sparse point-set vectorisation — [Automation in Construction 2025](https://www.sciencedirect.com/science/article/abs/pii/S0926580525000639)
- Multi-unit floor plan recognition (CAB) — [arXiv 2408.01526](https://arxiv.org/html/2408.01526v2)
- MitUNet — [arXiv 2512.02413](https://arxiv.org/pdf/2512.02413)
- CLIP-based recognition — [ASCE](https://ascelibrary.org/doi/10.1061/JCCEE5.CPENG-6500)
- RT-DETR CubiCasa detector — [GitHub](https://github.com/OldDeLorean/rtdetr-floorplan-detector)
- DoorDet — [arXiv 2508.07714](https://arxiv.org/pdf/2508.07714)
- SymPoint — [arXiv 2401.10556](https://arxiv.org/abs/2401.10556); SymPoint-V2 — [arXiv 2407.01928](https://arxiv.org/pdf/2407.01928)
- ArchCAD-400K — [arXiv 2503.22346](https://arxiv.org/pdf/2503.22346)
- CountGD — [project page](https://www.robots.ox.ac.uk/~vgg/research/countgd/)
- AECV-Bench — [arXiv 2601.04819](https://arxiv.org/pdf/2601.04819)
