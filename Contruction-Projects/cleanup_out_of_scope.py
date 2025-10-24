#!/usr/bin/env python3
"""
Cleanup script to filter out out-of-scope data for Construction Sensor Data Analysis Project

This script helps identify and optionally remove large data files that are out of scope
for the current analysis, keeping only essential files for version control.
"""

import os
import shutil
from pathlib import Path
import argparse

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def analyze_project_structure():
    """Analyze the current project structure and identify out-of-scope files"""
    
    print("=" * 80)
    print("🔍 ANALYZING PROJECT STRUCTURE")
    print("=" * 80)
    
    project_root = Path(".")
    
    # Define scope - what we want to keep vs remove
    scope_definition = {
        "KEEP": {
            "notebooks": "Jupyter notebooks with analysis code",
            "data/Becker_Keefe_2022/readme.txt": "Project documentation",
            "data/Becker_Keefe_2022/Becker_Keefe_DOI_2022.txt": "DOI reference",
            "data/Becker_Keefe_2022/Random_Forests/": "R analysis scripts",
            "data/Becker_Keefe_2022/Single_Tree_Data/": "Small tree data files",
            "*.md": "Documentation files",
            "*.py": "Python scripts",
            ".gitignore": "Git ignore file"
        },
        "REMOVE_OR_IGNORE": {
            "data/Becker_Keefe_2022/AS_TM_GNSS_Combined/*.csv": "Large raw sensor data (44M-226M each)",
            "data/processed/*.csv": "Generated processed data files",
            "plots/": "Generated visualization files",
            "*.png": "Generated plot images",
            "*.html": "Generated HTML outputs",
            "Miniforge3-Linux-x86_64.sh": "Large installer file"
        }
    }
    
    print("📋 SCOPE DEFINITION:")
    print("\n✅ KEEP (Essential for project):")
    for item, desc in scope_definition["KEEP"].items():
        print(f"   • {item}: {desc}")
    
    print("\n❌ REMOVE/IGNORE (Out of scope):")
    for item, desc in scope_definition["REMOVE_OR_IGNORE"].items():
        print(f"   • {item}: {desc}")
    
    return scope_definition

def find_large_files(threshold_mb=10):
    """Find files larger than threshold"""
    
    print(f"\n🔍 FINDING FILES LARGER THAN {threshold_mb}MB:")
    print("-" * 60)
    
    large_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.isfile(filepath):
                size_mb = get_file_size_mb(filepath)
                if size_mb > threshold_mb:
                    large_files.append((filepath, size_mb))
    
    # Sort by size (largest first)
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    total_size = 0
    for filepath, size_mb in large_files:
        print(f"   📁 {filepath}: {size_mb:.1f}MB")
        total_size += size_mb
    
    print(f"\n📊 TOTAL SIZE OF LARGE FILES: {total_size:.1f}MB")
    return large_files

def create_backup_directory():
    """Create backup directory for out-of-scope data"""
    backup_dir = Path("backup_out_of_scope")
    backup_dir.mkdir(exist_ok=True)
    return backup_dir

def move_out_of_scope_files(backup_dir, dry_run=True):
    """Move out-of-scope files to backup directory"""
    
    print(f"\n{'🔍 DRY RUN - ' if dry_run else '🚚 MOVING '}OUT-OF-SCOPE FILES:")
    print("-" * 60)
    
    # Files to move
    files_to_move = [
        "data/Becker_Keefe_2022/AS_TM_GNSS_Combined/",
        "data/processed/",
        "plots/",
        "Miniforge3-Linux-x86_64.sh"
    ]
    
    moved_files = []
    total_size_moved = 0
    
    for item in files_to_move:
        if os.path.exists(item):
            if os.path.isdir(item):
                # Move entire directory
                dest = backup_dir / item
                dest.parent.mkdir(parents=True, exist_ok=True)
                
                if not dry_run:
                    shutil.move(item, dest)
                    print(f"   📁 MOVED: {item} → {dest}")
                else:
                    print(f"   📁 WOULD MOVE: {item} → {dest}")
                
                # Calculate size
                size_mb = sum(get_file_size_mb(os.path.join(root, file)) 
                             for root, dirs, files in os.walk(item) 
                             for file in files)
                total_size_moved += size_mb
                moved_files.append((item, size_mb))
                
            elif os.path.isfile(item):
                # Move single file
                dest = backup_dir / item
                dest.parent.mkdir(parents=True, exist_ok=True)
                
                if not dry_run:
                    shutil.move(item, dest)
                    print(f"   📄 MOVED: {item} → {dest}")
                else:
                    print(f"   📄 WOULD MOVE: {item} → {dest}")
                
                size_mb = get_file_size_mb(item)
                total_size_moved += size_mb
                moved_files.append((item, size_mb))
    
    print(f"\n📊 TOTAL SIZE TO BE MOVED: {total_size_moved:.1f}MB")
    return moved_files

def create_data_summary():
    """Create a summary of the data that was moved"""
    
    summary_content = """# Data Summary - Construction Sensor Analysis

## 📊 Dataset Overview
- **Source**: Becker & Keefe (2022) - Excavator Activity Classification
- **Total Files**: 18 CSV files (6 days × 3 sampling rates)
- **Sampling Rates**: 10Hz, 20Hz, 50Hz
- **Total Size**: ~1.5GB (raw data)

## 🗂️ File Structure (Before Cleanup)
```
data/Becker_Keefe_2022/AS_TM_GNSS_Combined/
├── AS_TM_GNSS_8_10_10.csv  (44MB)
├── AS_TM_GNSS_8_10_20.csv  (86MB)
├── AS_TM_GNSS_8_10_50.csv  (214MB)
├── AS_TM_GNSS_8_11_10.csv  (46MB)
├── AS_TM_GNSS_8_11_20.csv  (91MB)
├── AS_TM_GNSS_8_11_50.csv  (226MB)
├── AS_TM_GNSS_9_20_10.csv  (35MB)
├── AS_TM_GNSS_9_20_20.csv  (71MB)
├── AS_TM_GNSS_9_20_50.csv  (177MB)
├── AS_TM_GNSS_9_21_10.csv  (35MB)
├── AS_TM_GNSS_9_21_20.csv  (68MB)
├── AS_TM_GNSS_9_21_50.csv  (169MB)
├── AS_TM_GNSS_9_26_10.csv  (35MB)
├── AS_TM_GNSS_9_26_20.csv  (69MB)
├── AS_TM_GNSS_9_26_50.csv  (170MB)
├── AS_TM_GNSS_9_27_10.csv  (37MB)
├── AS_TM_GNSS_9_27_20.csv  (73MB)
└── AS_TM_GNSS_9_27_50.csv  (184MB)
```

## 🎯 Analysis Scope
- **Primary Focus**: 50Hz data (highest resolution)
- **Days Analyzed**: 8/10, 8/11, 9/20 (3 days for main analysis)
- **All Days**: 8/10, 8/11, 9/20, 9/21, 9/26, 9/27 (6 days total)

## 📈 Data Characteristics
- **Time Range**: 4-6 hours per day
- **Rows per Day**: 750k-950k (50Hz data)
- **Total Rows**: 5M+ data points
- **Sampling Rate**: 50Hz (50 samples/second)

## 🔬 Sensor Data Fields
- **Accelerometer**: X, Y, Z (m/s²)
- **Linear Acceleration**: X, Y, Z (m/s²)
- **Gyroscope**: X, Y, Z (rad/s)
- **Sound**: Audio level (dB)
- **Element**: Activity labels (Clear, Move, Masticate, Travel, Delay, NULL, END)

## 📝 Notes
- Raw data files are large (35MB-226MB each) and not suitable for git
- Processed data and plots are generated from raw data
- Only essential files (notebooks, scripts, documentation) are kept in git
- Raw data can be restored from backup if needed
"""
    
    with open("DATA_SUMMARY.md", "w") as f:
        f.write(summary_content)
    
    print(f"\n📝 Created DATA_SUMMARY.md with project overview")

def main():
    parser = argparse.ArgumentParser(description="Cleanup out-of-scope data files")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be moved without actually moving files")
    parser.add_argument("--execute", action="store_true", 
                       help="Actually move the files (use with caution)")
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("❌ Please specify either --dry-run or --execute")
        print("   --dry-run: Show what would be moved")
        print("   --execute: Actually move the files")
        return
    
    # Analyze project structure
    scope_definition = analyze_project_structure()
    
    # Find large files
    large_files = find_large_files(threshold_mb=10)
    
    # Create backup directory
    backup_dir = create_backup_directory()
    
    # Move out-of-scope files
    dry_run = args.dry_run
    moved_files = move_out_of_scope_files(backup_dir, dry_run=dry_run)
    
    # Create data summary
    create_data_summary()
    
    print("\n" + "=" * 80)
    if dry_run:
        print("🔍 DRY RUN COMPLETED - No files were actually moved")
        print("   Use --execute to actually move the files")
    else:
        print("✅ CLEANUP COMPLETED - Out-of-scope files moved to backup/")
        print("   Check backup_out_of_scope/ directory for moved files")
    print("=" * 80)

if __name__ == "__main__":
    main()
