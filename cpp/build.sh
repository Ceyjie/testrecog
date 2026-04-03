#!/bin/bash
# build.sh - Build MedPal Robot Vision Server

set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "  MedPal Robot V2 - Build Script"
echo "=========================================="

# Create build directory
mkdir -p build
cd build

# Configure
echo "[1/3] Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "[2/3] Building..."
make -j$(nproc)

# Install
echo "[3/3] Installing..."
sudo make install

echo ""
echo "=========================================="
echo "  Build complete!"
echo "=========================================="
echo ""
echo "To run:"
echo "  cd /home/medpal/MedPalRobotV2"
echo "  ./cpp/build/vision_server"
echo ""
