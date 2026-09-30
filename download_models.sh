#!/bin/bash
# ============================================================================
# Download Models from GitHub Releases
# ============================================================================
# Usage: ./download_models.sh [OPTIONS]
#
# Examples:
#   ./download_models.sh                          # Download all models
#   ./download_models.sh --archive fp32           # Download FP32 only
#   ./download_models.sh --archive ptq            # Download INT8 PTQ only
#   ./download_models.sh --verify-only            # Only verify checksums
# ============================================================================

set -euo pipefail

# ---- Configuration ----
REPO="(redacted for blind review — see anonymized repo README)"
TAG="${TAG:-v0.1-alpha}"
BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"
MODELS_DIR="models"

# ---- Colors ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---- Functions ----
log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC} $1"; }

usage() {
    cat << EOF
Download Models from GitHub Releases

Usage: $0 [OPTIONS]

Options:
    --archive TYPE    Download specific archive only
                      Types: all, fp32, onnx, ptq
    --verify-only     Only verify checksums of existing files
    --output DIR      Output directory (default: models/)
    --tag TAG         Release tag (default: v0.1-alpha)
    --no-verify       Skip checksum verification
    -h, --help        Show this help

Examples:
    $0                          # Download all models
    $0 --archive fp32           # Download FP32 only
    $0 --archive ptq            # Download INT8 PTQ only
    $0 --verify-only            # Verify existing files
    $0 --output ./my_models     # Custom output directory

EOF
    exit 0
}

# ---- Parse Arguments ----
ARCHIVE="all"
VERIFY_ONLY=false
OUTPUT_DIR="$MODELS_DIR"
SKIP_VERIFY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --archive) ARCHIVE="$2"; shift 2 ;;
        --verify-only) VERIFY_ONLY=true; shift ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --no-verify) SKIP_VERIFY=true; shift ;;
        -h|--help) usage ;;
        *) log_err "Unknown option: $1"; usage ;;
    esac
done

# ---- Check Dependencies ----
check_deps() {
    local missing=()

    if ! command -v curl &>/dev/null && ! command -v wget &>/dev/null; then
        missing+=("curl or wget")
    fi
    if ! command -v tar &>/dev/null; then
        missing+=("tar")
    fi
    if [[ "$SKIP_VERIFY" == false ]]; then
        if ! command -v sha256sum &>/dev/null && ! command -v shasum &>/dev/null; then
            missing+=("sha256sum or shasum")
        fi
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_err "Missing dependencies:"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        exit 1
    fi
}

# ---- Download File ----
download_file() {
    local filename=$1
    local url="${BASE_URL}/${filename}"

    log_info "Downloading ${filename}..."

    if command -v curl &>/dev/null; then
        curl -L -# -o "$filename" "$url"
    elif command -v wget &>/dev/null; then
        wget --show-progress -O "$filename" "$url"
    fi

    if [[ $? -eq 0 ]]; then
        log_ok "Downloaded: ${filename}"
        return 0
    else
        log_err "Failed to download: ${filename}"
        return 1
    fi
}

# ---- Verify Checksum ----
verify_checksum() {
    local filename=$1
    local checksum_file="checksums.txt"

    if [[ ! -f "$checksum_file" ]]; then
        log_warn "Checksum file not found, skipping verification"
        return 0
    fi

    log_info "Verifying checksum for ${filename}..."

    local expected_hash
    expected_hash=$(grep "$filename" "$checksum_file" 2>/dev/null | awk '{print $1}')

    if [[ -z "$expected_hash" ]]; then
        log_warn "No checksum found for ${filename}, skipping"
        return 0
    fi

    local actual_hash
    if command -v sha256sum &>/dev/null; then
        actual_hash=$(sha256sum "$filename" | awk '{print $1}')
    else
        actual_hash=$(shasum -a 256 "$filename" | awk '{print $1}')
    fi

    if [[ "$actual_hash" == "$expected_hash" ]]; then
        log_ok "Checksum verified: ${filename}"
        return 0
    else
        log_err "Checksum mismatch for ${filename}!"
        echo "  Expected: $expected_hash"
        echo "  Got:      $actual_hash"
        return 1
    fi
}

# ---- Extract Archive ----
extract_archive() {
    local filename=$1
    local extract_dir=$2

    log_info "Extracting ${filename} to ${extract_dir}/..."

    mkdir -p "$extract_dir"
    tar -xzf "$filename" -C "$extract_dir"

    if [[ $? -eq 0 ]]; then
        log_ok "Extracted: ${filename}"
        return 0
    else
        log_err "Failed to extract: ${filename}"
        return 1
    fi
}

# ---- Show Models ----
show_models() {
    echo ""
    echo "============================================"
    echo "  Downloaded Models"
    echo "============================================"
    echo ""

    if [[ -d "$OUTPUT_DIR/fp32" ]]; then
        echo "📁 FP32 Models (PyTorch .pth):"
        ls -lh "$OUTPUT_DIR/fp32/"*.pth 2>/dev/null | awk '{print "  "$NF": "$5}'
        echo ""
    fi

    if [[ -d "$OUTPUT_DIR/onnx" ]]; then
        echo "📁 ONNX Models (FP32):"
        ls -lh "$OUTPUT_DIR/onnx/"*.onnx 2>/dev/null | awk '{print "  "$NF": "$5}'
        echo ""
    fi

    if [[ -d "$OUTPUT_DIR/ptq" ]]; then
        echo "📁 PTQ Models (INT8):"
        ls -lh "$OUTPUT_DIR/ptq/"*.onnx 2>/dev/null | awk '{print "  "$NF": "$5}'
        echo ""
    fi
}

# ---- Download All ----
download_all() {
    local files=(
        "models_all.tar.gz"
        "models_fp32.tar.gz"
        "models_onnx.tar.gz"
        "models_ptq.tar.gz"
        "checksums.txt"
    )

    for file in "${files[@]}"; do
        download_file "$file"
    done

    if [[ "$SKIP_VERIFY" == false ]]; then
        for file in "${files[@]}"; do
            if [[ "$file" != "checksums.txt" ]]; then
                verify_checksum "$file"
            fi
        done
    fi

    extract_archive "models_all.tar.gz" "$OUTPUT_DIR"
}

# ---- Download Specific Archive ----
download_specific() {
    local archive_type=$1
    local filename="models_${archive_type}.tar.gz"

    download_file "$filename"
    download_file "checksums.txt" 2>/dev/null || true

    if [[ "$SKIP_VERIFY" == false ]]; then
        verify_checksum "$filename"
    fi

    extract_archive "$filename" "$OUTPUT_DIR"
}

# ---- Main ----
main() {
    echo "============================================"
    echo "  Download Models from GitHub Releases"
    echo "  Tag: ${TAG}"
    echo "============================================"
    echo ""

    check_deps

    if [[ "$VERIFY_ONLY" == true ]]; then
        log_info "Verifying existing files..."
        cd "$OUTPUT_DIR"
        for file in models_*.tar.gz; do
            if [[ -f "$file" ]]; then
                verify_checksum "$file"
            fi
        done
        exit 0
    fi

    cd "$(dirname "$0")/.."

    case "$ARCHIVE" in
        all)    download_all ;;
        fp32)   download_specific "fp32" ;;
        onnx)   download_specific "onnx" ;;
        ptq)    download_specific "ptq" ;;
        *)      log_err "Unknown archive type: $ARCHIVE"; usage ;;
    esac

    show_models

    echo ""
    log_ok "Download complete!"
    echo ""
    echo "Usage examples:"
    echo "  # Load PyTorch model"
    echo "  import torch"
    echo "  model = torch.load('${OUTPUT_DIR}/fp32/mobilenetv2.pth')"
    echo ""
    echo "  # Use ONNX model"
    echo "  import onnxruntime as ort"
    echo "  session = ort.InferenceSession('${OUTPUT_DIR}/onnx/mobilenetv2_fp32.onnx')"
}

main
