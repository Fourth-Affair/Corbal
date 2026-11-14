#!/usr/bin/env bash
# Deploy the app to a single server (e.g. a Verda CPU instance) you created by hand. See deploy/README.md.
#
#   deploy/deploy.sh <ip> all       # first time: setup + push + up
#   deploy/deploy.sh <ip> setup     # Docker, firewall (22/80/443), fail2ban (once per server)
#   deploy/deploy.sh <ip> push      # code (tracked files of the working tree), CubiCasa weights, .env, warm cache
#   deploy/deploy.sh <ip> up        # build + (re)start the stack
#   deploy/deploy.sh <ip> update    # push + up: deploy a new version
#   deploy/deploy.sh <ip> status    # container health + site URL
#   deploy/deploy.sh <ip> logs [service]   # follow logs (caddy | frontend | backend)
#   deploy/deploy.sh <ip> down      # stop the stack
#   deploy/deploy.sh <ip> ssh       # shell in /srv/stoa
#
# <ip> can be left out when DEPLOY_HOST is set in the environment or in .env.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
REMOTE_DIR=/srv/stoa
WEIGHTS_SRC=${WEIGHTS_SRC:-dataset/cubicasa-pretrained/model_best_val_loss_var.pkl}
COMPOSE="docker compose -f compose.prod.yaml"

env_value() { [ -f .env ] && sed -n "s/^$1=//p" .env | tail -1 | tr -d "\"'" || true; }

COMMANDS=" setup push up update all status logs down ssh "
if [[ $# -ge 1 && $COMMANDS != *" $1 "* ]]; then
  HOST=$1; shift
else
  HOST=${DEPLOY_HOST:-$(env_value DEPLOY_HOST)}
fi
CMD=${1:-}; shift || true
[[ -n $HOST && -n $CMD ]] || { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30)
remote() { ssh "${SSH_OPTS[@]}" "root@$HOST" "$@"; }
site_address() { local s; s=$(env_value SITE_ADDRESS); echo "${s:-${HOST//./-}.sslip.io}"; }

cmd_setup() {
  echo "==> setup $HOST"
  remote 'bash -s' < deploy/bootstrap.sh
}

cmd_push() {
  echo "==> push code to $HOST:$REMOTE_DIR"
  local untracked
  untracked=$(git ls-files --others --exclude-standard -- backend frontend deploy models compose.prod.yaml)
  if [[ -n $untracked ]]; then
    echo "warning: untracked files are NOT deployed (git add them):"; echo "$untracked" | sed 's/^/  /'
  fi
  # Tracked files including uncommitted edits: `git stash create` makes a commit of the working tree without
  # touching it (empty output when clean -> HEAD).
  local tree; tree=$(git stash create); tree=${tree:-HEAD}
  echo "    $(git rev-parse --short HEAD)$( [[ $tree != HEAD ]] && echo ' + uncommitted changes')"
  # Replace the code; keep .env, dataset/ (cache, uploads) and the CubiCasa weights.
  git archive "$tree" | remote "set -e; mkdir -p $REMOTE_DIR && cd $REMOTE_DIR
    find . -mindepth 1 -maxdepth 1 ! -name .env ! -name dataset ! -name models -exec rm -rf {} +
    [ -d models ] && find models -mindepth 1 ! -name cubicasa-net.pkl -exec rm -rf {} + || true
    tar -x -C $REMOTE_DIR"

  if remote "test -s $REMOTE_DIR/models/cubicasa-net.pkl"; then
    echo "==> CubiCasa weights already on the server"
  elif [[ -f $WEIGHTS_SRC ]]; then
    echo "==> upload CubiCasa weights (208 MB)"
    rsync -aP -e "ssh ${SSH_OPTS[*]}" "$WEIGHTS_SRC" "root@$HOST:$REMOTE_DIR/models/cubicasa-net.pkl"
  else
    echo "warning: $WEIGHTS_SRC not found: the image pipeline will be unavailable (set WEIGHTS_SRC)"
  fi

  [[ -f .env ]] || { echo "error: no .env (copy .env.example and fill it in)"; exit 1; }
  echo "==> .env (without VERDA_* / DEPLOY_HOST), SITE_ADDRESS=$(site_address)"
  { grep -v -e '^VERDA_' -e '^DEPLOY_HOST=' -e '^SITE_ADDRESS=' .env; echo "SITE_ADDRESS=$(site_address)"; } |
    remote "umask 077; cat > $REMOTE_DIR/.env"

  local warm=()
  for d in dataset/cache/agent dataset/cache/graphs; do [[ -d $d ]] && warm+=("$d"); done
  if (( ${#warm[@]} )); then
    echo "==> warm cache: ${warm[*]}"
    remote "mkdir -p $REMOTE_DIR/dataset/cache"
    rsync -a --ignore-existing -e "ssh ${SSH_OPTS[*]}" "${warm[@]}" "root@$HOST:$REMOTE_DIR/dataset/cache/"
  fi
}

cmd_up() {
  echo "==> build + start on $HOST (first build ~5-10 min)"
  remote "cd $REMOTE_DIR && $COMPOSE up -d --build --remove-orphans && docker image prune -f >/dev/null"
  cmd_status
}

cmd_status() {
  remote "cd $REMOTE_DIR && $COMPOSE ps"
  local url="https://$(site_address)"
  echo "==> $url"
  if curl -fsS -o /dev/null -m 20 --retry 6 --retry-delay 5 --retry-all-errors "$url"; then
    echo "    site is up"
  else
    echo "    not answering yet (backend health / certificate can take a minute): deploy/deploy.sh $HOST logs"
  fi
}

case $CMD in
  setup)  cmd_setup ;;
  push)   cmd_push ;;
  up)     cmd_up ;;
  update) cmd_push; cmd_up ;;
  all)    cmd_setup; cmd_push; cmd_up ;;
  status) cmd_status ;;
  logs)   remote "cd $REMOTE_DIR && $COMPOSE logs -f --tail 200 $*" ;;
  down)   remote "cd $REMOTE_DIR && $COMPOSE down" ;;
  ssh)    ssh -t "${SSH_OPTS[@]}" "root@$HOST" "cd $REMOTE_DIR; exec bash -l" ;;
  *)      echo "unknown command: $CMD"; exit 1 ;;
esac
