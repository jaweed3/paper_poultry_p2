#!/bin/bash
# ============================================================================
# Upload Models to GitHub Releases (Fixed Version)
# ============================================================================
# Usage: ./scripts/upload_models_to_release.sh [--tag TAG] [--dry-run]
# ============================================================================

set -euo pipefail

# ---- Configuration ----
REPO="(redacted for blind review — see anonymized repo README)"
TAG="${TAG:-v0.1-alpha}"
RELEASE_NAME="Model Checkpoints (Alpha)"
MODELS_DIR="models"

# ---- Colors ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ---- Functions ----
log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: $0 [--tag TAG] [--dry-run]"
    echo ""
    echo "Options:"
    echo "  --tag TAG       Release tag (default: v0.1-alpha)"
    echo "  --dry-run       Only create archives, don't upload"
    echo "  -h, --help      Show this help"
    exit 0
}

# ---- Parse Arguments ----
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag) TAG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) log_err "Unknown option: $1"; usage ;;
    esac
done

# ---- Preflight Checks ----
check_deps() {
    local missing=()

    if ! command -v gh &>/dev/null; then
        missing+=("gh (GitHub CLI)")
    fi
    if ! command -v tar &>/dev/null; then
        missing+=("tar")
    fi
    if ! command -v gzip &>/dev/null; then
        missing+=("gzip")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_err "Missing dependencies:"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "Install GitHub CLI: https://cli.github.com/"
        exit 1
    fi

    # Check if logged in
    if ! gh auth status &>/dev/null; then
        log_err "Not logged in to GitHub CLI. Run: gh auth login"
        exit 1
    fi
}

# ---- Create Archives ----
create_archives() {
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local archive_dir="releases_${timestamp}"
    mkdir -p "$archive_dir"

    log_info "Creating model archives..."

    # 1. All models
    log_info "Packing all models (fp32 + onnx + ptq)..."
    tar -czf "${archive_dir}/models_all.tar.gz" \
        -C "$MODELS_DIR" \
        fp32/ onnx/ ptq/
    log_ok "models_all.tar.gz created"

    # 2. FP32 only
    log_info "Packing FP32 models..."
    tar -czf "${archive_dir}/models_fp32.tar.gz" \
        -C "$MODELS_DIR" \
        fp32/
    log_ok "models_fp32.tar.gz created"

    # 3. ONNX only
    log_info "Packing ONNX models..."
    tar -czf "${archive_dir}/models_onnx.tar.gz" \
        -C "$MODELS_DIR" \
        onnx/
    log_ok "models_onnx.tar.gz created"

    # 4. PTQ (INT8) only
    log_info "Packing PTQ (INT8) models..."
    tar -czf "${archive_dir}/models_ptq.tar.gz" \
        -C "$MODELS_DIR" \
        ptq/
    log_ok "models_ptq.tar.gz created"

    # 5. Generate checksums
    log_info "Generating SHA256 checksums..."
    cd "$archive_dir"
    sha256sum models_*.tar.gz > checksums.txt
    cd ..
    log_ok "checksums.txt created"

    # Show sizes
    log_info "Archive sizes:"
    ls -lh "$archive_dir"/*.tar.gz | awk '{print "  "$NF": "$5}'

    echo "$archive_dir"
}

# ---- Upload to GitHub ----
upload_release() {
    local archive_dir=$1

    log_info "Creating GitHub release: ${TAG}..."

    # Delete existing release if exists
    gh release delete "$TAG" --yes --cleanup-tag 2>/dev/null || true

    # Create release with body file
    local body_file="${archive_dir}/release_body.md"
    cat > "$body_file" << 'EOF'
## Model Checkpoints

### Available Archives

| Archive | Contents | Size |
|---------|----------|------|
| `models_all.tar.gz` | FP32 + ONNX + PTQ | ~59 MB |
| `models_fp32.tar.gz` | FP32 only (.pth) | ~27 MB |
| `models_onnx.tar.gz` | ONNX FP32 only | ~27 MB |
| `models_ptq.tar.gz` | PTQ INT8 only | ~5.5 MB |

### Quick Download

```bash
# Download all models
echo "Model archive: see anonymized repo release page (URL redacted for blind review)"
tar -xzf models_all.tar.gz

# Verify checksum
sha256sum -c checksums.txt
```

### Models Included

- **MobileNetV2** - 97.83% accuracy, 8.48 MB (FP32)
- **ShuffleNetV2** - 97.38% accuracy, 4.89 MB (FP32)
- **EfficientNet-B0** - 98.35% accuracy, 15.30 MB (FP32)

### Quantization Notes

- INT8 PTQ causes accuracy drop in EfficientNet-B0 (98.35% → 35.46%)
- MobileNetV2 and ShuffleNetV2 are robust to INT8 quantization
EOF

    # Create release
    gh release create "$TAG" \
        --repo "$REPO" \
        --title "$RELEASE_NAME" \
        --notes-file "$body_file"

    log_ok "Release created, now uploading assets..."

    # Upload files one by one
    local files=(
        "${archive_dir}/models_all.tar.gz"
        "${archive_dir}/models_fp32.tar.gz"
        "${archive_dir}/models_onnx.tar.gz"
        "${archive_dir}/models_ptq.tar.gz"
        "${archive_dir}/checksums.txt"
    )

    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            local filename
            filename=$(basename "$file")
            log_info "Uploading ${filename}..."
            gh release upload "$TAG" "$file" --repo "$REPO" --clobber
            log_ok "Uploaded: ${filename}"
        fi
    done

    log_ok "All assets uploaded!"
    log_info "Release URL: https://github.com/${REPO}/releases/tag/${TAG}"
}

# ---- Cleanup ----
cleanup() {
    local archive_dir=$1
    log_info "Cleaning up temporary files..."
    rm -rf "$archive_dir"
    log_ok "Cleanup complete"
}

# ---- Main ----
main() {
    echo "============================================"
    echo "  Upload Models to GitHub Releases"
    echo "============================================"
    echo ""

    check_deps

    local archive_dir
    archive_dir=$(create_archives)

    if [[ "$DRY_RUN" == true ]]; then
        log_warn "Dry run mode - skipping upload"
        log_info "Archives saved in: ${archive_dir}/"
        log_info "Run without --dry-run to upload"
    else
        upload_release "$archive_dir"
        cleanup "$archive_dir"
    fi

    echo ""
    log_ok "Done!"
}

main
