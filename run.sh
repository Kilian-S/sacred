#!/bin/bash
# ==============================================================================
# SACRED ONE-CLICK TRAINING & TENSORBOARD MONITOR
# ==============================================================================
#
# Usage:
#   ./run.sh [--episodes N] [--switch-every M] [--batch-size B] [--hidden-dim H]
#
# Example:
#   ./run.sh --episodes 100 --switch-every 10 --batch-size 64
#
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Define color codes for a premium terminal UI
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Get directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo -e "${CYAN}${BOLD}"
echo "======================================================================"
echo "          SACRED: Reinforcement Learning Training & Monitor           "
echo "======================================================================"
echo -e "${NC}"

# Check for virtual environment
if [ -d ".venv" ]; then
    echo -e "${GREEN}[✔] Local virtual environment (.venv) detected.${NC}"
    # Activate virtual environment
    source .venv/bin/activate
else
    echo -e "${YELLOW}[!] Warning: No .venv directory found in the current directory.${NC}"
    echo -e "    Attempting to run using the system python..."
fi

# Ensure launcher script is executable
chmod +x scripts/run_and_monitor.py

echo -e "${BLUE}[i] Running the Python launcher and opening TensorBoard...${NC}"
echo -e "    Any extra parameters you passed will be forwarded to the training script."
echo ""

# Run the python monitor launcher, passing all script arguments
PYTHONPATH=. python3 scripts/run_and_monitor.py "$@"
