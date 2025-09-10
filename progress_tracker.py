#!/usr/bin/env python3
"""
Progress Tracker Utility
Provides real-time progress tracking for policy extraction pipeline
"""

import time
import sys
from typing import List, Optional
from datetime import datetime, timedelta

class ProgressTracker:
    """Real-time progress tracker for policy extraction pipeline"""
    
    def __init__(self, total_items: int, stage_name: str = "Processing"):
        self.total_items = total_items
        self.stage_name = stage_name
        self.current_item = 0
        self.start_time = time.time()
        self.stage_start_time = time.time()
        self.successful = 0
        self.failed = 0
        self.current_file = ""
        self.substage = ""
        
    def update(self, item_name: str = "", substage: str = "", success: Optional[bool] = None):
        """Update progress with current item and optional substage"""
        if success is not None:
            if success:
                self.successful += 1
            else:
                self.failed += 1
        
        if item_name:
            self.current_item += 1
            self.current_file = item_name
        
        if substage:
            self.substage = substage
            
        self._display_progress()
    
    def set_stage(self, stage_name: str):
        """Change to a new processing stage"""
        self.stage_name = stage_name
        self.stage_start_time = time.time()
        self.substage = ""
        self._display_progress()
    
    def _display_progress(self):
        """Display current progress"""
        # Calculate timing
        elapsed = time.time() - self.start_time
        stage_elapsed = time.time() - self.stage_start_time
        
        # Calculate ETA
        if self.current_item > 0:
            avg_time_per_item = elapsed / self.current_item
            remaining_items = self.total_items - self.current_item
            eta_seconds = remaining_items * avg_time_per_item
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "Calculating..."
        
        # Progress percentage
        progress_pct = (self.current_item / self.total_items) * 100 if self.total_items > 0 else 0
        
        # Progress bar
        bar_length = 30
        filled_length = int(bar_length * progress_pct / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Build status line
        status_parts = []
        if self.substage:
            status_parts.append(f"{self.substage}")
        if self.current_file:
            # Truncate long filenames
            filename = self.current_file[:40] + "..." if len(self.current_file) > 43 else self.current_file
            status_parts.append(f"📄 {filename}")
        
        status = " | ".join(status_parts) if status_parts else "Initializing..."
        
        # Clear line and print progress
        print(f"\r\033[K", end="")  # Clear current line
        print(f"🔄 {self.stage_name}: [{bar}] {progress_pct:.1f}% ({self.current_item}/{self.total_items})", end="")
        print(f" | ✅ {self.successful} ❌ {self.failed} | ⏱️ {eta} | {status}", end="", flush=True)
    
    def complete(self, message: str = ""):
        """Mark processing as complete"""
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print(f"\r\033[K", end="")  # Clear current line
        print(f"✅ {self.stage_name} Complete!")
        print(f"   📊 Processed: {self.current_item}/{self.total_items} files")
        print(f"   ✅ Successful: {self.successful}")
        print(f"   ❌ Failed: {self.failed}")
        print(f"   ⏱️ Total time: {elapsed_str}")
        if message:
            print(f"   📝 {message}")
        print()

class MultiStageTracker:
    """Tracker for multi-stage pipeline processing"""
    
    def __init__(self, files: List[str]):
        self.files = files
        self.total_files = len(files)
        self.current_stage = 1
        self.total_stages = 2  # Chunking + Policy Extraction
        self.stage_names = ["Smart Chunking", "Policy Extraction"]
        self.overall_start = time.time()
        self.current_tracker: Optional[ProgressTracker] = None
        
        # File complexity assessment
        self.file_complexities = self._assess_file_complexities()
        
        print("🚀 Starting Policy Extraction Pipeline")
        print("=" * 60)
        print(f"📁 Processing folder: ADDA - DGE")
        print(f"📄 Total files: {self.total_files}")
        print(f"🔄 Stages: {' → '.join(self.stage_names)}")
        self._display_file_summary()
        print("=" * 60)
        print()
    
    def _assess_file_complexities(self) -> dict:
        """Assess complexity of each file based on size"""
        import os
        complexities = {}
        
        for file_path in self.files:
            try:
                size = os.path.getsize(file_path)
                if size < 500_000:  # < 500KB
                    complexity = "Simple"
                elif size < 2_000_000:  # < 2MB
                    complexity = "Medium"
                elif size < 4_000_000:  # < 4MB
                    complexity = "Complex"
                else:
                    complexity = "Very Complex"
                
                complexities[file_path] = {
                    "complexity": complexity,
                    "size_mb": size / 1_000_000
                }
            except:
                complexities[file_path] = {"complexity": "Unknown", "size_mb": 0}
        
        return complexities
    
    def _display_file_summary(self):
        """Display summary of files to be processed"""
        complexity_counts = {}
        for file_info in self.file_complexities.values():
            complexity = file_info["complexity"]
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        print("📊 File Complexity Assessment:")
        for complexity, count in complexity_counts.items():
            print(f"   {complexity}: {count} files")
    
    def start_stage(self, stage_num: int):
        """Start a new processing stage"""
        self.current_stage = stage_num
        stage_name = self.stage_names[stage_num - 1]
        
        print(f"\n🔄 Stage {stage_num}/{self.total_stages}: {stage_name}")
        print("-" * 40)
        
        self.current_tracker = ProgressTracker(self.total_files, stage_name)
    
    def update_progress(self, item_name: str = "", substage: str = "", success: Optional[bool] = None):
        """Update current stage progress"""
        if self.current_tracker:
            # Extract just filename for display
            if item_name:
                filename = item_name.split("/")[-1] if "/" in item_name else item_name
                self.current_tracker.update(filename, substage, success)
            else:
                self.current_tracker.update(item_name, substage, success)
    
    def complete_stage(self, message: str = ""):
        """Complete current stage"""
        if self.current_tracker:
            self.current_tracker.complete(message)
    
    def complete_pipeline(self, total_policies_extracted: int = 0):
        """Complete entire pipeline"""
        total_elapsed = time.time() - self.overall_start
        elapsed_str = str(timedelta(seconds=int(total_elapsed)))
        
        print("🎉 Pipeline Complete!")
        print("=" * 60)
        print(f"⏱️ Total processing time: {elapsed_str}")
        print(f"📄 Files processed: {self.total_files}")
        if total_policies_extracted > 0:
            print(f"📋 Policies extracted: {total_policies_extracted}")
        print(f"💾 Output location: output/ADDA - DGE/")
        print("=" * 60)

