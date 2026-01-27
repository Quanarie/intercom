#!/bin/bash
# Voice Intercom Streamlit App - Startup Script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       🎤 Voice Intercom - Streamlit Application           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}📍 Working directory: $SCRIPT_DIR${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found!${NC}"
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source venv/bin/activate || . venv/Scripts/activate

# Check if requirements are installed
echo -e "${YELLOW}📦 Checking dependencies...${NC}"
pip install -q -r requirements.txt

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models/)" ]; then
    echo -e "${RED}❌ Error: No trained models found in ./models/${NC}"
    echo -e "${RED}   Please ensure model checkpoints (*.pth files) are in the models directory${NC}"
    exit 1
fi

MODEL_COUNT=$(ls models/*.pth 2>/dev/null | wc -l)
echo -e "${GREEN}✓ Found $MODEL_COUNT trained models${NC}"

# Start Streamlit app
echo -e ""
echo -e "${GREEN}✓ Starting Streamlit application...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🌐 Open your browser to: http://localhost:8501${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e ""

streamlit run app.py --client.toolbarMode=viewer

# Deactivate on exit
deactivate 2>/dev/null || true
