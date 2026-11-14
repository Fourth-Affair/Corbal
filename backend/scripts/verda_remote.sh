#!/usr/bin/env bash
# Runs ON the Verda VM (started by scripts/verda_train.py). Builds the detector training set from the original
# CubiCasa5K release, trains, exports ONNX. Writes $ROOT/DONE or $ROOT/FAILED when it stops.
#   env: BASE EPOCHS IMGSZ BATCH
set -euo pipefail
ROOT=$HOME/stoa
trap 'echo "FAILED at line $LINENO"; touch "$ROOT/FAILED"' ERR
cd "$ROOT"
rm -f DONE FAILED

if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH=$HOME/.local/bin:$PATH
if [ ! -d .venv-train ]; then
  uv venv -q --python 3.12 .venv-train
  # PyPI torch wheels are CUDA builds on Linux
  VIRTUAL_ENV=$ROOT/.venv-train uv pip install -q torch torchvision "ultralytics>=8.3,<9" onnx onnxslim onnxruntime \
    pydantic pillow numpy
fi
PY=$ROOT/.venv-train/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'; print('GPU:', torch.cuda.get_device_name(0))"

# Original CubiCasa5K (images + official split files); the datacenter link is much faster than uploading
mkdir -p data
if [ ! -d data/orig/cubicasa5k ]; then
  curl -sS -L -C - --retry 5 -o data/cubicasa5k.zip "https://zenodo.org/api/records/2613548/files/cubicasa5k.zip/content"
  echo "0ce0b203d1e3c125b51087b219bd23b9  data/cubicasa5k.zip" | md5sum -c -
  $PY -c "import zipfile; zipfile.ZipFile('data/cubicasa5k.zip').extractall('data/orig')"
  rm data/cubicasa5k.zip
fi

export CUBICASA_ORIG_DIR=$ROOT/data/orig/cubicasa5k
export CACHE_DIR=$ROOT/data/cache            # cache/svg = ground-truth labels, uploaded from the laptop
export DATASET_DIR=$ROOT/data/junction-dataset  # absent here: images come from the original release
cd "$ROOT/backend"
$PY -m app.cubicasa.export_yolo --holdout "$ROOT/data/holdout.jsonl" --out "$ROOT/data/yolo-det"
$PY -m app.cubicasa.train --data "$ROOT/data/yolo-det/data.yaml" --base "$BASE" --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" --batch "$BATCH" --device 0 --workers 8 --onnx --out "$ROOT/data/out/symbol-det.pt"
cp "$ROOT"/data/runs/symdet/results.csv "$ROOT/data/out/symbol-det.results.csv"
echo "imgsz=$IMGSZ base=$BASE epochs=$EPOCHS" > "$ROOT/data/out/symbol-det.info"
touch "$ROOT/DONE"
echo DONE
