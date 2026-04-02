#!/bin/bash
# Voice Chat — Whisper STT → RRR Inference → Piper TTS
# Runs on GPD Pocket 4, uses built-in mic + speakers

WHISPER_MODEL="$HOME/tools/whisper/ggml-base.en.bin"
PIPER_MODEL="$HOME/tools/piper/en_GB-jenny_dioco-medium.onnx"
ENGINE_DIR="$HOME/projects/recursive-routing-racer-rs"
TMPDIR="/tmp/voice_chat"

mkdir -p "$TMPDIR"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== RRR Voice Chat ===${NC}"
echo -e "${CYAN}Engine: Phi-4 3.8B | Vulkan | 6.6 tok/s${NC}"
echo -e "${CYAN}STT: Whisper base.en | TTS: Piper Jenny${NC}"
echo ""
echo -e "${YELLOW}Press ENTER to speak, ENTER again to stop recording.${NC}"
echo -e "${YELLOW}Type 'quit' to exit.${NC}"
echo ""

# Build engine if needed
cd "$ENGINE_DIR"
if [ ! -f target/release/rrr ]; then
    echo "Building engine..."
    source ~/.cargo/env
    cargo build --release 2>/dev/null
fi

while true; do
    # Wait for user to press enter
    echo -e "${GREEN}[READY]${NC} Press ENTER to speak..."
    read -r input

    if [ "$input" = "quit" ]; then
        echo "Goodbye!"
        break
    fi

    # Record audio until enter is pressed again
    echo -e "${CYAN}[LISTENING]${NC} Recording... press ENTER to stop"
    pw-record --format s16 --rate 16000 --channels 1 "$TMPDIR/input.wav" &
    RECORD_PID=$!
    read -r
    kill $RECORD_PID 2>/dev/null
    wait $RECORD_PID 2>/dev/null

    # Transcribe with Whisper
    echo -e "${CYAN}[THINKING]${NC} Transcribing..."
    TRANSCRIPT=$(whisper-cpp -m "$WHISPER_MODEL" -f "$TMPDIR/input.wav" --no-timestamps -t 4 2>/dev/null | grep -v '^\[' | tr -d '\n' | sed 's/^ *//')

    if [ -z "$TRANSCRIPT" ]; then
        echo -e "${YELLOW}[EMPTY]${NC} Didn't catch that, try again."
        continue
    fi

    echo -e "${CYAN}[HEARD]${NC} \"$TRANSCRIPT\""

    # Run inference
    echo -e "${CYAN}[GENERATING]${NC}"
    source ~/.cargo/env
    RESPONSE=$(echo "$TRANSCRIPT" | timeout 120 cargo run --release 2>/dev/null | grep -v '^\[' | grep -v '^===' | grep -v '^$' | head -20)

    if [ -z "$RESPONSE" ]; then
        RESPONSE="I didn't generate a response. Try again."
    fi

    echo -e "${GREEN}[RESPONSE]${NC} $RESPONSE"

    # Speak with Piper
    echo -e "${CYAN}[SPEAKING]${NC}"
    echo "$RESPONSE" | piper --model "$PIPER_MODEL" --output_file "$TMPDIR/response.wav" 2>/dev/null
    pw-play "$TMPDIR/response.wav" 2>/dev/null

    echo ""
done
