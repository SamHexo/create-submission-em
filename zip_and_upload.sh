#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: ${0##*/} <nom-sous-dossier | chemin-absolu>" >&2
  echo "  Zippe le dossier puis envoie l’archive sur Emily (dossier projet: codes)." >&2
  echo "  Nom seul → grade-mle-task-agent-em/workspace/codes/<nom>" >&2
  echo "  Chemin absolu → ce dossier tel quel." >&2
  exit 1
}

# Liste les noms de fichiers déjà présents dans le dossier projet « codes » (sortie humaine d’emily).
emily_list_codes_filenames() {
  local page=1 page_size=500
  local out
  while true; do
    out=$(emily project files list --page-size "$page_size" --page "$page")
    echo "$out" | awk '
      /^  [^[:space:]]/ { name=$0; sub(/^  /, "", name) }
      /^    Folder: / {
        if ($2 == "codes" && length(name)) print name
        name=""
      }
    '
    if echo "$out" | grep -Fq "More files available"; then
      page=$((page + 1))
    else
      break
    fi
  done
}

# Premier nom disponible : <stem>.zip, puis <stem>2.zip, <stem>3.zip, …
pick_upload_basename() {
  local stem="$1"
  local -a existing
  mapfile -t existing < <(emily_list_codes_filenames)
  local name n
  name="${stem}.zip"
  for e in "${existing[@]}"; do
    [[ "$e" == "$name" ]] || continue
    name=""
    break
  done
  if [[ -n "$name" ]]; then
    printf '%s\n' "$name"
    return
  fi
  n=2
  while true; do
    name="${stem}${n}.zip"
    local taken=0
    for e in "${existing[@]}"; do
      [[ "$e" == "$name" ]] && taken=1 && break
    done
    if [[ "$taken" -eq 0 ]]; then
      printf '%s\n' "$name"
      return
    fi
    n=$((n + 1))
    if [[ "$n" -gt 9999 ]]; then
      echo "Erreur: plus de 9998 variantes déjà présentes pour ${stem}.zip sur Emily." >&2
      exit 1
    fi
  done
}

[[ $# -eq 1 ]] || usage
[[ -n "$1" ]] || usage

arg="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
codes_dir="$script_dir/grade-mle-task-agent-em/workspace/codes"

if [[ "$arg" == /* ]]; then
  if [[ ! -d "$arg" ]]; then
    echo "Erreur: dossier inexistant: $arg" >&2
    exit 1
  fi
  target="$(cd "$arg" && pwd)"
  zip_parent="$(dirname "$target")"
  folder_name="$(basename "$target")"
else
  if [[ "$arg" == *"/"* || "$arg" == *".."* ]]; then
    echo "Erreur: sans chemin absolu, fournir uniquement le nom du dossier sous codes/." >&2
    exit 1
  fi
  if [[ ! -d "$codes_dir" ]]; then
    echo "Erreur: dossier codes introuvable: $codes_dir" >&2
    exit 1
  fi
  target="$codes_dir/$arg"
  if [[ ! -d "$target" ]]; then
    echo "Erreur: sous-dossier inexistant: $target" >&2
    exit 1
  fi
  zip_parent="$codes_dir"
  folder_name="$arg"
fi

upload_basename=$(pick_upload_basename "$folder_name")
zip_path="${TMPDIR:-/tmp}/${upload_basename}"
cleanup() { rm -f "$zip_path"; }
trap cleanup EXIT

(cd "$zip_parent" && zip -r "$zip_path" "$folder_name")

emily project files upload --folder codes "$zip_path"
