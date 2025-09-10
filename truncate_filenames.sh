#!/bin/bash

# Check if directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Define directory path from argument
DIR="$1"

# Function to sanitize filename
sanitize_filename() {
    local filename="$1"
    # Remove leading underscores
    filename="${filename#_}"
    # Replace spaces with underscores
    filename="${filename// /_}"
    # Remove any non-alphanumeric characters except dots and underscores
    filename=$(echo "$filename" | sed 's/[^a-zA-Z0-9._-]//g')
    echo "$filename"
}

# Function to truncate filename
truncate_filename() {
    local file="$1"
    local basename=$(basename "$file")
    local dirname=$(dirname "$file")
    local extension="${basename##*.}"
    local filename="${basename%.*}"
    
    # Skip numbered files (e.g., 16.pdf, 17.pdf)
    if [[ $filename =~ ^[0-9]+$ ]]; then
        return
    fi
    
    # Sanitize the filename
    filename=$(sanitize_filename "$filename")
    
    # Truncate to max 36 chars (40 - 1 for dot - 3 for extension)
    local newfilename="${filename:0:36}.${extension}"
    
    # Only rename if needed and if the new name is different
    if [ "$basename" != "$newfilename" ]; then
        echo "Renaming: $basename -> $newfilename"
        mv "$file" "$dirname/$newfilename"
    fi
}

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist"
    exit 1
fi

# Process all PDF files in the directory
cd "$DIR"
for file in *.pdf; do
    if [ -f "$file" ]; then
        truncate_filename "$file"
    fi
done

echo "Filename truncation complete!"
