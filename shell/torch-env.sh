# torch-env shell helpers.
# Source from your ~/.bashrc, ~/.zshrc, or a devcontainer postCreateCommand:
#   source /path/to/torch-env/shell/torch-env.sh

# Activate the newest torch-YYYY.MM.N env installed locally.
torch-latest() {
  local tag
  tag=$(conda env list \
        | awk '{print $1}' \
        | grep -E '^torch-20[0-9]{2}\.[0-9]{1,2}\.[0-9]+$' \
        | sed 's/^torch-//' \
        | sort -V \
        | tail -1)
  [ -z "$tag" ] && { echo "no torch-YYYY.MM.* envs installed"; return 1; }
  conda activate "torch-$tag"
}

# Activate the env this repo pins via .torch-env-<plat>.
torch-repo() {
  local plat
  case "$(uname -s)" in
    Darwin) plat=mac ;;
    Linux)  plat=linux ;;
    *) echo "unsupported platform: $(uname -s)"; return 1 ;;
  esac

  local root file tag
  root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "not in a git repo"; return 1; }
  file="$root/.torch-env-$plat"
  [ -f "$file" ] || { echo "no $file"; return 1; }

  tag=$(tr -d '[:space:]' < "$file")
  if ! conda env list | awk '{print $1}' | grep -qx "torch-$tag"; then
    echo "env torch-$tag not installed. Build it from the torch-env repo: make $tag"
    return 1
  fi
  conda activate "torch-$tag"
}
