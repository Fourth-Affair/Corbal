# floorprep

Two small, self-contained building blocks for floor-plan images:

1. **`preprocess(img)`**: makes scans, photos and CAD exports look alike before they go to the CubiCasa model.
2. **`detect_penmarks(img)`**: a yes/no answer to "does this drawing have hand-written annotations?". Use it to decide which drawings a (more expensive) model should look at to interpret the marks.

Everything runs locally on CPU with open-source models. Copy this folder into a project and install `requirements.txt`.

```python
from floorprep import load_image, preprocess, detect_penmarks

img = load_image("plan.png")
res = preprocess(img)            # res.images: model-ready image(s), res.info: what was done
if detect_penmarks(img):         # True / False
    ...                          # send the whole original image to the interpretation model
```

In this backend it runs as the first step of `POST /api/plans/{plan_id}/analyze` (see `app/main.py`):
the upload is preprocessed (no split), the models analyse the preprocessed image, and the result
(`PlanGraph`) carries `has_penmarks` and `preprocessing` for later steps. Set `PREPROCESS=off` to skip it.

## Preprocessing steps

| Step | What it does | Why |
|---|---|---|
| orientation fix (optional, `orient=True`) | Runs OCR on all 8 rotations/flips and keeps the one where the text reads best. Images with no readable text are left alone. | Many scans are upside down, sideways or mirrored. This is the slowest step (OCR). |
| resize | Long side to 1000–1500 px (upscales tiny images, downscales 4000 px scans). | Consistent scale for the model and the following steps. |
| background whitening | Estimates the paper colour (ignoring lines and large fills) and divides it out. | Removes grey photocopy wash, uneven lighting and colour casts. Grey balcony fills survive. |
| despeckle (optional) | Whites out tiny isolated dark specks. | Scan and photocopy noise. |
| crop / split | Finds the drawing(s), drops small clusters (logos, disclaimers) and margins. Multi-plan sheets become separate images. | More resolution for the plan, one floor per model call. |
| pad | White border, size a multiple of 32. | Network stride. |

**Use the cropped images as model input only.** For question answering and the interpretation model, keep the *original*: the crop can remove title blocks, scale notes ("ei mittakaavassa"), legends and north arrows.

`res.info` records the transforms, so model outputs can be mapped back to the original:

- `orientation` (`flip`, `rot90`): applied first
- `scale`: applied next
- `crops`: `[x, y, w, h]` boxes in that oriented + resized image
- `timing`: seconds per stage

All parameters live in `floorprep.preprocess.DEFAULTS`. Override any of them per call with `preprocess(img, params={...})`.

## Pen-mark detection

1. **Candidates:**
   - thin, irregular strokes in any colour that isn't the plan's own print colour
   - every text box found by RapidOCR's detector, which also finds black pen and pencil
2. **Scoring:** each candidate is scored by CLIP (ViT-B-32, OpenAI weights): handwritten vs printed text, logos, drawing lines and fills.
3. **Flag:** `True` when any candidate is confidently handwritten (P ≥ `penmarks.THRESHOLD` = 0.85).

`penmark_score(img)` returns the underlying 0–1 score if you want to move the threshold.

## Known limits

- **No labelled evaluation yet.** The threshold was set from a manual review of 48 crops from the CubiCasa5K sample.
- **Possible false flags:**
  - upside-down or very blurry printed labels (the detector runs on the original orientation)
  - hatching
  - coloured line fragments
- **Old hand-drafted drawings** have hand-lettered labels that look like handwriting, so they can be flagged even without later annotations.
- **First run downloads the CLIP weights** (about 350 MB) from the Hugging Face hub.
