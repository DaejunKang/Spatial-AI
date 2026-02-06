#!/bin/bash
# ============================================================================
# 3D Reconstruction External Dependencies Setup
#
# 이 스크립트는 3DGS 및 3DGUT 모델의 외부 의존성을 설치합니다.
#
# External Repositories:
#   - gaussian-splatting (graphdeco-inria): 3D Gaussian Splatting 레퍼런스 구현
#   - gsplat (nerfstudio-project): NVIDIA 3DGUT 통합 CUDA 가속 래스터라이저
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="$SCRIPT_DIR/external"

echo "============================================"
echo "  3D Reconstruction - External Setup"
echo "============================================"
echo ""

# ============================================================================
# Step 1: Git Submodule 초기화
# ============================================================================
echo "[Step 1] Initializing git submodules..."
cd "$SCRIPT_DIR/../../.."  # workspace root
git submodule update --init --recursive
echo "  Done."
echo ""

# ============================================================================
# Step 2: 3DGS (gaussian-splatting) 의존성 설치
# ============================================================================
echo "[Step 2] Setting up 3DGS (gaussian-splatting)..."
GS_DIR="$EXTERNAL_DIR/gaussian-splatting"

if [ -f "$GS_DIR/train.py" ]; then
    echo "  Repository found: $GS_DIR"

    # 3DGS 서브모듈 (diff-gaussian-rasterization, simple-knn)
    cd "$GS_DIR"
    git submodule update --init --recursive 2>/dev/null || true

    # Python 의존성
    echo "  Installing 3DGS Python dependencies..."
    pip install plyfile tqdm 2>/dev/null || true

    # diff-gaussian-rasterization (CUDA 필요)
    if [ -d "$GS_DIR/submodules/diff-gaussian-rasterization" ]; then
        echo "  Building diff-gaussian-rasterization..."
        pip install "$GS_DIR/submodules/diff-gaussian-rasterization" 2>/dev/null || {
            echo "  Warning: Failed to build diff-gaussian-rasterization (CUDA required)"
        }
    fi

    # simple-knn (CUDA 필요)
    if [ -d "$GS_DIR/submodules/simple-knn" ]; then
        echo "  Building simple-knn..."
        pip install "$GS_DIR/submodules/simple-knn" 2>/dev/null || {
            echo "  Warning: Failed to build simple-knn (CUDA required)"
        }
    fi

    echo "  3DGS setup complete."
else
    echo "  Warning: 3DGS repository not found at $GS_DIR"
    echo "  Run: git submodule update --init --recursive"
fi
echo ""

# ============================================================================
# Step 3: gsplat (NVIDIA 3DGUT) 설치
# ============================================================================
echo "[Step 3] Setting up gsplat (NVIDIA 3DGUT)..."
GSPLAT_DIR="$EXTERNAL_DIR/gsplat"

if [ -f "$GSPLAT_DIR/setup.py" ]; then
    echo "  Repository found: $GSPLAT_DIR"

    # gsplat 설치 (JIT compile on first run)
    echo "  Installing gsplat..."
    pip install -e "$GSPLAT_DIR" 2>/dev/null || {
        echo "  Warning: Failed to install gsplat from source"
        echo "  Trying pip install gsplat..."
        pip install gsplat 2>/dev/null || {
            echo "  Warning: Could not install gsplat. CUDA compilation required."
        }
    }

    # gsplat example 의존성
    if [ -f "$GSPLAT_DIR/examples/requirements.txt" ]; then
        echo "  Installing gsplat example dependencies..."
        pip install -r "$GSPLAT_DIR/examples/requirements.txt" 2>/dev/null || true
    fi

    echo "  gsplat (3DGUT) setup complete."
else
    echo "  Warning: gsplat repository not found at $GSPLAT_DIR"
    echo "  Run: git submodule update --init --recursive"
fi
echo ""

# ============================================================================
# Step 4: 공통 의존성
# ============================================================================
echo "[Step 4] Installing common dependencies..."
pip install torch torchvision 2>/dev/null || true
pip install numpy Pillow tqdm 2>/dev/null || true
echo "  Done."
echo ""

# ============================================================================
# 상태 확인
# ============================================================================
echo "============================================"
echo "  Installation Status"
echo "============================================"

# 3DGS
if [ -f "$GS_DIR/train.py" ]; then
    echo "  [OK] 3DGS (gaussian-splatting)"
else
    echo "  [MISSING] 3DGS (gaussian-splatting)"
fi

# gsplat
if python -c "import gsplat; print(f'  [OK] gsplat v{gsplat.__version__}')" 2>/dev/null; then
    :
else
    if [ -f "$GSPLAT_DIR/gsplat/__init__.py" ]; then
        echo "  [PARTIAL] gsplat (source available, not pip-installed)"
    else
        echo "  [MISSING] gsplat"
    fi
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Usage:"
echo "  # 3DGS (Approach 1):"
echo "  python reconstruction/approach1_3dgs.py /path/to/data"
echo ""
echo "  # 3DGUT (Approach 2):"
echo "  python reconstruction/approach2_3dgut.py /path/to/data"
echo ""
