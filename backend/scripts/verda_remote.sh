#!/usr/bin/env bash
# Runs ON the Verda VM (started by scripts/verda_train.py). Writes $ROOT/DONE or $ROOT/FAILED when it stops.
#   TASK=symdet  (default) build detector training sets from the original CubiCasa5K release, then for each config
#                in SWEEP (presets below; "current" = the shipped detector, evaluation only): train, export, and
#                evaluate (det_eval on val+test, det caches for the fit table, fusion refit fit-on val and train+val)
#                env: SWEEP EPOCHS BATCH FRACTION QUESTIONS_ZIP QUESTIONS_URL       -> data/out/<config>/...
#   TASK=wallseg train the wall segmenter on uploaded label masks (data/cache/wallmasks, from `wallseg prepare`)
#                env: EPOCHS BATCH                                 -> data/out/wallseg/wallseg.{pt,log.csv}
set -euo pipefail
ROOT=$HOME/stoa
trap 'echo "FAILED at line $LINENO"; touch "$ROOT/FAILED"' ERR
cd "$ROOT"
rm -f DONE FAILED

if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH=$HOME/.local/bin:$PATH
if [ ! -d .venv-train ]; then
  uv venv -q --python 3.12 .venv-train
  # PyPI torch wheels are CUDA builds on Linux. sklearn/numpy/joblib pinned to the laptop's versions: the fusion
  # models fitted here are loaded there.
  VIRTUAL_ENV=$ROOT/.venv-train uv pip install -q torch torchvision "ultralytics==8.4.212" onnx onnxslim onnxruntime \
    pydantic pillow "numpy==2.5.4" "scikit-learn==1.9.1" "joblib==1.6.0" albumentations remotezip
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
export DATASET_DIR=$ROOT/data/junction-dataset  # only questions.jsonl here: images come from the original release
cd "$ROOT/backend"

if [ "${TASK:-symdet}" = wallseg ]; then
  $PY -c "import segmentation_models_pytorch" 2>/dev/null || \
    VIRTUAL_ENV=$ROOT/.venv-train uv pip install -q segmentation-models-pytorch scipy
  test -f "$CACHE_DIR/wallmasks/index.json"  # uploaded by verda_train.py
  mkdir -p "$ROOT/data/out/wallseg"
  MODELS_DIR=$ROOT/data/out/wallseg $PY -u -m app.cubicasa.wallseg train --device cuda --epochs "$EPOCHS" \
    --batch "$BATCH" --workers 18
  touch "$ROOT/DONE"
  echo DONE
  exit 0
fi

# Challenge questions (labels for the fusion refit): just the one file out of the dataset zip. Called right before
# each refit, so a zip still downloading on the VM is picked up once complete; never fatal.
# (QUESTIONS_ZIP: a zip already on the VM, e.g. ~/junction-dataset.zip; else QUESTIONS_URL with range requests)
fetch_questions() {
  [ -f "$DATASET_DIR/questions.jsonl" ] && return 0
  local zip=${QUESTIONS_ZIP:-}
  zip=${zip/#\~/$HOME}
  if [ -n "$zip" ] && [ -f "$zip" ]; then
    $PY scripts/fetch_questions.py "$zip" "$DATASET_DIR/questions.jsonl" || echo "could not read $zip (still downloading?)"
  elif [ -n "${QUESTIONS_URL:-}" ]; then
    $PY scripts/fetch_questions.py "$QUESTIONS_URL" "$DATASET_DIR/questions.jsonl" || echo "could not fetch questions"
  fi
  return 0
}

# name -> base imgsz tile rot90 scale
preset() {
  case "$1" in
    A) echo "yolo11s.pt 1280 0 0 0.3" ;;     # the shipped recipe, retrained: run-to-run noise
    B) echo "yolo11s.pt 1600 0 1 0.3" ;;     # higher resolution + 8 orientations
    C) echo "yolo11m.pt 1600 0 1 0.3" ;;     # bigger model
    D) echo "yolo11m.pt 1024 1024 1 0.5" ;;  # native-resolution tiles
    *) echo "unknown preset $1" >&2; return 1 ;;
  esac
}

# Evaluate one detector: det_eval (val picks thresholds, test reported), det caches, fusion refits
evaluate() {
  local name=$1 weights=$2 out=$ROOT/data/out/$1
  mkdir -p "$out"
  export SYMDET_WEIGHTS=$weights SYMDET_CACHE_DIR=$ROOT/data/det/$name
  echo "== eval $name ($weights)"
  $PY -u -m app.cubicasa.det_eval --splits val test --out "$out/det_eval.json" --viz 8 --viz-dir "$out/viz" \
    | tee "$out/det_eval.txt"
  fetch_questions
  if [ -f "$DATASET_DIR/questions.jsonl" ] && [ -f "$CACHE_DIR/features_table.jsonl" ]; then
    $PY -u -m app.cubicasa.symdet --splits train val test
    for fit_on in val train+val; do
      mkdir -p "$out/fit-$fit_on"
      MODELS_DIR=$out/fit-$fit_on $PY -u -m app.cubicasa.fit --test --final --fit-on "$fit_on" | tee "$out/fit-$fit_on/fit.txt"
    done
  else
    echo "no questions.jsonl / features_table.jsonl: skipping the fusion refit"
  fi
  unset SYMDET_WEIGHTS SYMDET_CACHE_DIR
}

for name in ${SWEEP:-A}; do
  # a rerun skips training a config whose weights exist (evaluation reruns: detections are cached)
  if [ "$name" = current ]; then
    evaluate current "$(ls "$ROOT"/data/current/symbol-det.pt "$ROOT"/data/current/symbol-det.onnx 2>/dev/null | head -1)"
    continue
  fi
  read -r base imgsz tile rot scale < <(preset "$name")
  if [ -f "$ROOT/data/out/$name/symbol-det.pt" ]; then
    echo "== $name: weights exist, evaluating only"
    evaluate "$name" "$ROOT/data/out/$name/symbol-det.pt"
    continue
  fi
  data=$ROOT/data/yolo-det-t$tile-r$rot
  if [ ! -f "$data/data.yaml" ]; then
    $PY -u -m app.cubicasa.export_yolo --holdout "$ROOT/data/holdout.jsonl" --out "$data" --tile "$tile" \
      $([ "$rot" = 1 ] && echo --rot90) --workers 20
  fi
  echo "== train $name: base=$base imgsz=$imgsz tile=$tile rot90=$rot scale=$scale"
  $PY -u -m app.cubicasa.train --data "$data/data.yaml" --base "$base" --epochs "${EPOCHS:-80}" --imgsz "$imgsz" \
    --batch "${BATCH:-0.7}" --fraction "${FRACTION:-1}" --scale "$scale" --tile "$tile" --name "$name" --device 0 \
    --workers 16 --onnx --out "$ROOT/data/out/$name/symbol-det.pt"
  cp "$ROOT/data/runs/$name/results.csv" "$ROOT/data/out/$name/results.csv"
  evaluate "$name" "$ROOT/data/out/$name/symbol-det.pt"
  # tiles are large and only this config uses them
  if [ "$tile" != 0 ]; then rm -rf "$data"; fi
done

touch "$ROOT/DONE"
echo DONE
