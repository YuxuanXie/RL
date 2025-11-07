#!/usr/bin/env python3
"""
Extract point cloud chunks and layouts from zip files.
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from zipfile import ZipFile


def extract_zip(zip_path, dest_dir, description=""):
    """Extract a zip file to destination directory (flattened)."""
    if description:
        print(f"Extracting {description}")
    
    with ZipFile(zip_path, 'r') as zip_ref:
        # Extract all files to destination, flattening directory structure
        for member in zip_ref.namelist():
            # Skip directories
            if member.endswith('/'):
                continue
            
            # Get just the filename (no path)
            filename = os.path.basename(member)
            
            # Extract to destination
            source = zip_ref.open(member)
            target_path = os.path.join(dest_dir, filename)
            
            with open(target_path, 'wb') as target:
                target.write(source.read())


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract point cloud chunks and layouts from zip files.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--rm-zips', '-rm',
        action='store_true',
        help='Remove zip files after extraction (default: keep them)'
    )
    
    args = parser.parse_args()
    
    # Check for point cloud chunks
    chunk_pattern = 'pcd/chunk_*.zip'
    chunk_files = sorted(glob.glob(chunk_pattern))
    
    if not chunk_files:
        print("No chunk zip files found in pcd folder!")
        sys.exit(1)
    
    # Extract point clouds
    for chunk_file in chunk_files:
        chunk_name = os.path.basename(chunk_file)
        print(f"Extracting {chunk_name}")
        extract_zip(chunk_file, 'pcd/')
    
    print("")
    print("All point clouds extracted!")
    
    # Extract layouts
    layout_zip = 'layout/layout.zip'
    if os.path.exists(layout_zip):
        print("Extracting layouts")
        extract_zip(layout_zip, 'layout/')
        print("All layouts extracted!")
    else:
        print(f"Warning: {layout_zip} not found, skipping...")
    
    # Interactive prompt to remove zip files
    remove_zips = args.rm_zips
    
    if not remove_zips:
        print("")
        try:
            response = input("Do you want to remove the zip files? (Y/N): ")
            if response.strip().upper() in ['Y', 'YES']:
                remove_zips = True
        except (EOFError, KeyboardInterrupt):
            print("\nKeeping zip files.")
            remove_zips = False
    
    # Remove zip files if requested
    if remove_zips:
        print("Removing zip files...")
        
        # Remove chunk files
        for chunk_file in chunk_files:
            try:
                os.remove(chunk_file)
            except OSError as e:
                print(f"Error removing {chunk_file}: {e}")
        
        # Remove layout file
        if os.path.exists(layout_zip):
            try:
                os.remove(layout_zip)
            except OSError as e:
                print(f"Error removing {layout_zip}: {e}")
        
        print("Zip files removed!")
    else:
        print("Zip files kept.")


if __name__ == '__main__':
    main()
