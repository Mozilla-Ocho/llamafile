#!/bin/bash

# Default values
VERSION=""
SOURCE_DIR=""
DEST_DIR=""

# Function to display help
show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo
  echo "Script to copy and rename binary files with version numbers."
  echo
  echo "Options:"
  echo "  -v, --version VERSION     Version number to append to filenames (required)"
  echo "  -s, --source DIR          Source directory containing the binaries (required)"
  echo "  -d, --dest DIR            Destination directory for renamed binaries (required)"
  echo "  -h, --help                Display this help message and exit"
  echo
  echo "Example:"
  echo "  $0 -v 0.9.2 -s /llamafile-release/0.9.2/llamafile-0.9.2/bin -d /llamafile-release/0.9.2/release"
  echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--version)
      VERSION="$2"
      shift 2
      ;;
    -s|--source)
      SOURCE_DIR="$2"
      shift 2
      ;;
    -d|--dest)
      DEST_DIR="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Error: Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if required arguments are provided
if [ -z "$VERSION" ] || [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ]; then
  echo "Error: Missing required arguments"
  show_help
  exit 1
fi

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source directory does not exist: $SOURCE_DIR"
  exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# List of binaries to copy and rename
BINARIES=(
  "llamafile"
  "llamafile-bench"
  "llamafiler"
  "sdfile"
  "localscore"
  "whisperfile"
  "zipalign"
)

# Copy and rename each binary
for binary in "${BINARIES[@]}"; do
  if [ -f "${SOURCE_DIR}/${binary}" ]; then
    cp "${SOURCE_DIR}/${binary}" "${DEST_DIR}/${binary}-${VERSION}"
    echo "Copied ${binary} to ${DEST_DIR}/${binary}-${VERSION}"
  else
    echo "Warning: ${SOURCE_DIR}/${binary} not found"
  fi
done

# Make all binaries in destination directory executable
chmod +x "${DEST_DIR}"/*