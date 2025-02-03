#!/bin/bash

# Create ~/.llamafile directory if it doesn't exist
LLAMAFILE_DIR="$HOME/.llamafile"
mkdir -p "$LLAMAFILE_DIR"

# Define model names and URLs as a simple map using strings
# Format: "friendly_name|url"
MODELS=(
    "LLaVA 1.5|https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/resolve/main/llava-v1.5-7b-q4.llamafile?download=true"
    "TinyLlama-1.1B|https://huggingface.co/Mozilla/TinyLlama-1.1B-Chat-v1.0-llamafile/resolve/main/TinyLlama-1.1B-Chat-v1.0.F16.llamafile?download=true"
    "Mistral-7B-Instruct|https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q4_0.llamafile?download=true"
    "Phi-3-mini-4k-instruct|https://huggingface.co/Mozilla/Phi-3-mini-4k-instruct-llamafile/resolve/main/Phi-3-mini-4k-instruct.F16.llamafile?download=true"
    "Mixtral-8x7B-Instruct|https://huggingface.co/Mozilla/Mixtral-8x7B-Instruct-v0.1-llamafile/resolve/main/mixtral-8x7b-instruct-v0.1.Q5_K_M.llamafile?download=true"
    "WizardCoder-Python-34B|https://huggingface.co/Mozilla/WizardCoder-Python-34B-V1.0-llamafile/resolve/main/wizardcoder-python-34b-v1.0.Q5_K_M.llamafile?download=true"
    "WizardCoder-Python-13B|https://huggingface.co/jartine/wizardcoder-13b-python/resolve/main/wizardcoder-python-13b.llamafile?download=true"
    "LLaMA-3-Instruct-70B|https://huggingface.co/Mozilla/Meta-Llama-3-70B-Instruct-llamafile/resolve/main/Meta-Llama-3-70B-Instruct.Q4_0.llamafile?download=true"
    "LLaMA-3-Instruct-8B|https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile/resolve/main/Meta-Llama-3-8B-Instruct.Q5_K_M.llamafile?download=true"
    "Rocket-3B|https://huggingface.co/Mozilla/rocket-3B-llamafile/resolve/main/rocket-3b.Q5_K_M.llamafile?download=true"
    "OLMo-7B|https://huggingface.co/Mozilla/OLMo-7B-0424-llamafile/resolve/main/OLMo-7B-0424.Q6_K.llamafile?download=true"
    "E5-Mistral-7B-Instruct|https://huggingface.co/Mozilla/e5-mistral-7b-instruct/resolve/main/e5-mistral-7b-instruct-Q5_K_M.llamafile?download=true"
    "mxbai-embed-large-v1|https://huggingface.co/Mozilla/mxbai-embed-large-v1-llamafile/resolve/main/mxbai-embed-large-v1-f16.llamafile?download=true"
)

# Function to create a safe filename from model name
create_safe_filename() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' ' '-'
}

# Function to get URL for a given model
get_model_url() {
    local model_name="$1"
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name url <<< "$entry"
        if [ "$name" = "$model_name" ]; then
            echo "$url"
            return 0
        fi
    done
    echo ""
    return 1
}

# Function to get model name by index
get_model_by_index() {
    local index=$1
    if [ "$index" -ge 1 ] && [ "$index" -le "${#MODELS[@]}" ]; then
        IFS='|' read -r name _ <<< "${MODELS[$((index-1))]}"
        echo "$name"
        return 0
    fi
    echo ""
    return 1
}

# Function to display help message
show_help() {
    echo "Usage: $0 [OPTIONS] [MODEL_NAME]"
    echo
    echo "This tool helps to download llamafile models and run them on your system."
    echo "Models are sourced from: https://github.com/Mozilla-Ocho/llamafile"
    echo "Models are stored in: $LLAMAFILE_DIR"
    echo
    echo "Options:"
    echo "  --help     Show this help message"
    echo "  --list     List available models"
    echo "  --installed List installed models"
    echo
    echo "Example:"
    echo "  $0               # Interactive model selection"
    echo "  $0 --installed   # List installed models"
    echo "  $0 \"Mistral 7B Instruct\"    # Download/run specific model"
}

# Function to list installed models
list_installed() {
    echo "Installed models in $LLAMAFILE_DIR:"
    echo "-----------------------------------"
    local found=0
    for file in "$LLAMAFILE_DIR"/*.llamafile; do
        if [ -f "$file" ]; then
            echo "  - $(basename "$file" .llamafile | tr '-' ' ' | sed 's/\b\(.\)/\u\1/g')"
            found=1
        fi
    done
    if [ $found -eq 0 ]; then
        echo "No models installed yet."
    fi
}

# Function to list available models with numbering
list_models() {
    local list_only=$1
    echo "Available models:"
    local i=1
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name url <<< "$entry"
            if [ "$list_only" = "true" ]; then
                echo "  - $name"
            else
                echo "  $i) $name"
            fi
        ((i++))
    done
}

# Function to download and run model
download_and_run() {
    local model_name="$1"
    local url=$(get_model_url "$model_name")
    local filename="$(create_safe_filename "$model_name").llamafile"
    local filepath="$LLAMAFILE_DIR/$filename"

    if [ -f "$filepath" ]; then
        echo "Model '$model_name' is already installed."
        echo "Options:"
        echo "1) Run existing model"
        echo "2) Download again"
        echo "3) Cancel"
        read -p "Choose an option (1-3): " choice
        
        case $choice in
            1)
                echo "Running existing model..."
                chmod a+x "$filepath"
                "$filepath"
                exit 0
                ;;
            2)
                echo "Re-downloading model..."
                ;;
            3)
                echo "Operation cancelled."
                exit 0
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac
    fi

    echo "Downloading $model_name to $filepath..."
    wget -O "$filepath" "$url"

    if [ $? -ne 0 ]; then
        echo "Error: Download failed"
        rm -f "$filepath"
        exit 1
    fi

    echo "Making file executable..."
    chmod a+x "$filepath"

    echo "Running $model_name..."
    "$filepath"
}

# Main script logic
case "$1" in
    --help)
        show_help
        exit 0
        ;;
    --list)
        list_models "true"
        exit 0
        ;;
    --installed)
        list_installed
        exit 0
        ;;
    "")
        echo "Welcome to LLaMA model downloader!"
        echo "Models will be stored in: $LLAMAFILE_DIR"
        echo
        list_models "false"
        echo
        read -p "Select a model number (1-${#MODELS[@]}): " selection
        MODEL_NAME=$(get_model_by_index "$selection")
        if [ -z "$MODEL_NAME" ]; then
            echo "Invalid selection"
            exit 1
        fi
        ;;
    *)
        MODEL_NAME="$1"
        ;;
esac

if [ -z "$(get_model_url "$MODEL_NAME")" ]; then
    echo "Error: Model '$MODEL_NAME' not found"
    echo "Use --list to see available models"
    exit 1
fi

download_and_run "$MODEL_NAME"