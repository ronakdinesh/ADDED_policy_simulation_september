#!/bin/bash

# Define paths
DED_FOLDER="/Users/karthik/Projects/Github/doc-agent-policy/Docs/DED"
OUTPUT_DIR="Output"

# Check if Python environment is active
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
  echo "Warning: No Python virtual environment detected. It's recommended to run this in a virtual environment."
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# Step 1: Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Step 2: Run smart chunking agent on DED folder
echo "Running b_smart_chunking_agent.py on DED folder..."
# Without the --force flag, it will skip already processed files
python b_smart_chunking_agent.py --input "$DED_FOLDER" --output "$OUTPUT_DIR"

# Step 3: Wait for completion and run policy agent
echo "Running c_policy_agent.py on processed files..."
python c_policy_agent.py --input "$OUTPUT_DIR" --output "$OUTPUT_DIR/extracted_policies.xlsx"

echo "Processing complete! Results saved to $OUTPUT_DIR/extracted_policies.xlsx" 