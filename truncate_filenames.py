#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

def clean_filename(filename):
    # Keep only the filename, remove path
    filename = os.path.basename(filename)
    
    # Remove Arabic characters (U+0600 to U+06FF)
    filename = re.sub(r'[\u0600-\u06FF]+', '', filename)
    
    # Remove multiple spaces, replace with single space
    filename = re.sub(r'\s+', ' ', filename)
    
    # Keep only alphanumeric characters, spaces, dots, and hyphens
    filename = re.sub(r'[^a-zA-Z0-9\s\.-]', '', filename)
    
    # Remove spaces before extension
    name, ext = os.path.splitext(filename)
    name = name.strip()
    
    # Remove leading/trailing spaces and dots
    name = re.sub(r'^[\s\.]+|[\s\.]+$', '', name)
    
    # If filename becomes empty after cleaning, use a default name with original extension
    if not name:
        name = "document"
    
    # Truncate to reasonable length (max 100 chars for name + extension)
    if len(name) > 100:
        name = name[:97] + "..."
    
    return f"{name}{ext}"

def process_directory(directory):
    directory_path = Path(directory)
    
    # Store original and new filenames
    file_mappings = {}
    
    # First pass: Generate new filenames
    for file_path in directory_path.glob("*.pdf"):
        original_name = file_path.name
        new_name = clean_filename(original_name)
        
        # Ensure no duplicate filenames
        counter = 1
        base_new_name = new_name
        while new_name in file_mappings.values():
            name, ext = os.path.splitext(base_new_name)
            new_name = f"{name}_{counter}{ext}"
            counter += 1
        
        file_mappings[original_name] = new_name
    
    # Second pass: Rename files
    for original_name, new_name in file_mappings.items():
        original_path = directory_path / original_name
        new_path = directory_path / new_name
        
        try:
            # Create backup of original file
            backup_path = directory_path / f"{original_name}.bak"
            shutil.copy2(original_path, backup_path)
            
            # Rename file
            os.rename(original_path, new_path)
            print(f"Renamed: {original_name} -> {new_name}")
            
        except Exception as e:
            print(f"Error processing {original_name}: {str(e)}")
            # Restore from backup if available
            if backup_path.exists():
                shutil.copy2(backup_path, original_path)
            continue
        
        # Remove backup after successful rename
        if backup_path.exists():
            os.remove(backup_path)

if __name__ == "__main__":
    directory = "./Docs/ADAFSA_Batch 1"
    print(f"Processing files in: {directory}")
    process_directory(directory)
    print("Done!") 