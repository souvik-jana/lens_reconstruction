#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
skills=(
  gwemfish-simulate
  gwemfish-infer
  gwemfish-cfg
  gwemfish-plot
  gwemfish-batch
  gwemfish-pal
  pal-infer
  pal-plot
  pal-sim
  lenstronomy-infer
  lenstronomy-sim
  lensing-mock
)
agents=(gwemfish.md gwemfish-batch.md)

mkdir -p "$HOME/.cursor/skills" "$HOME/.cursor/agents"

link_dir() {
  src="$1"
  dest="$2"
  if [[ ! -e "$src" ]]; then
    echo "missing: $src" >&2
    exit 1
  fi
  if [[ -e "$dest" && ! -L "$dest" ]]; then
    echo "skip $dest (exists and is not a symlink; remove manually to link)" >&2
    return 0
  fi
  ln -sfn "$src" "$dest"
  echo "linked $dest -> $src"
}

for name in "${skills[@]}"; do
  link_dir "$repo_root/.cursor/skills/$name" "$HOME/.cursor/skills/$name"
done

for name in "${agents[@]}"; do
  link_dir "$repo_root/.cursor/agents/$name" "$HOME/.cursor/agents/$name"
done

echo
echo "Global Cursor skills/agents linked from $repo_root"
echo "Machine-local (NOT symlinked — copy once per machine and edit paths):"
echo "  cp -r $repo_root/.cursor/skills/gwemfish-local.example ~/.cursor/skills/gwemfish-local"
echo "  cp -r $repo_root/.cursor/skills/pal-local.example ~/.cursor/skills/pal-local"
