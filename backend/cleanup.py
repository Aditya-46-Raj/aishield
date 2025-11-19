"""
Compliance Cleanup Script for AIShield KYC System

This script performs automated cleanup of old audit logs and evidence files
according to the retention policy defined in compliance_config.py.

Features:
- Removes logs and evidence older than retention period
- Archives old logs before deletion (optional)
- Secure file deletion with overwriting
- Dry-run mode for testing
- Detailed logging of cleanup operations

Usage:
    python cleanup.py                 # Normal cleanup
    python cleanup.py --dry-run       # Show what would be deleted without deleting
    python cleanup.py --force          # Skip confirmation prompts
    python cleanup.py --archive        # Archive logs before deletion

Author: AIShield Team
Date: November 15, 2025
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import compliance configuration
try:
    from compliance_config import (
        COMPLIANCE_CONFIG,
        get_retention_days,
        get_log_retention_days
    )
except ImportError:
    logger.error("Could not import compliance_config.py")
    sys.exit(1)

# Paths
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "outputs"
ARCHIVE_DIR = BASE_DIR / COMPLIANCE_CONFIG.get("archive_path", "logs/archive")

def get_file_age_days(file_path: Path) -> int:
    """
    Get the age of a file in days
    
    Args:
        file_path: Path to the file
        
    Returns:
        Age in days
    """
    try:
        modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        age = datetime.now() - modified_time
        return age.days
    except Exception as e:
        logger.warning(f"Could not get age for {file_path}: {e}")
        return 0

def secure_delete_file(file_path: Path):
    """
    Securely delete a file by overwriting before deletion
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if COMPLIANCE_CONFIG.get("secure_file_deletion", False):
            # Overwrite file with random data
            file_size = file_path.stat().st_size
            with open(file_path, 'wb') as f:
                f.write(os.urandom(file_size))
        
        # Delete file
        file_path.unlink()
        return True
    except Exception as e:
        logger.error(f"Failed to delete {file_path}: {e}")
        return False

def archive_log_file(log_file: Path, archive_dir: Path) -> bool:
    """
    Archive a log file before deletion
    
    Args:
        log_file: Path to log file
        archive_dir: Directory to archive to
        
    Returns:
        True if successful
    """
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / log_file.name
        shutil.copy2(log_file, archive_path)
        logger.info(f"  📦 Archived: {log_file.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to archive {log_file}: {e}")
        return False

def cleanup_old_logs(dry_run=False, archive=False):
    """
    Clean up old audit logs based on retention policy
    
    Args:
        dry_run: If True, don't actually delete files
        archive: If True, archive logs before deletion
        
    Returns:
        Tuple of (deleted_count, archived_count, freed_bytes)
    """
    logger.info("=" * 70)
    logger.info("CLEANING UP OLD AUDIT LOGS")
    logger.info("=" * 70)
    
    if not LOGS_DIR.exists():
        logger.warning(f"Logs directory does not exist: {LOGS_DIR}")
        return 0, 0, 0
    
    retention_days = get_log_retention_days()
    logger.info(f"Retention policy: {retention_days} days")
    
    deleted_count = 0
    archived_count = 0
    freed_bytes = 0
    
    # Find old log files
    for log_file in LOGS_DIR.glob("*.json"):
        age_days = get_file_age_days(log_file)
        
        if age_days > retention_days:
            file_size = log_file.stat().st_size
            logger.info(f"  🗑️  {log_file.name} (age: {age_days} days, size: {file_size/1024:.2f} KB)")
            
            if not dry_run:
                # Archive if requested
                if archive and COMPLIANCE_CONFIG.get("archive_old_logs", False):
                    if archive_log_file(log_file, ARCHIVE_DIR):
                        archived_count += 1
                
                # Delete file
                if secure_delete_file(log_file):
                    deleted_count += 1
                    freed_bytes += file_size
            else:
                deleted_count += 1  # Count for dry run
                freed_bytes += file_size
    
    logger.info(f"\nDeleted: {deleted_count} files")
    if archived_count > 0:
        logger.info(f"Archived: {archived_count} files")
    logger.info(f"Freed: {freed_bytes/1024:.2f} KB")
    
    return deleted_count, archived_count, freed_bytes

def cleanup_old_evidence(dry_run=False):
    """
    Clean up old evidence files (images, videos, heatmaps)
    
    Args:
        dry_run: If True, don't actually delete files
        
    Returns:
        Tuple of (deleted_count, freed_bytes)
    """
    logger.info("\n" + "=" * 70)
    logger.info("CLEANING UP OLD EVIDENCE FILES")
    logger.info("=" * 70)
    
    if not OUTPUTS_DIR.exists():
        logger.warning(f"Outputs directory does not exist: {OUTPUTS_DIR}")
        return 0, 0
    
    retention_days = get_retention_days()
    logger.info(f"Retention policy: {retention_days} days")
    
    deleted_count = 0
    freed_bytes = 0
    
    # Extensions to clean up
    extensions = [".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi"]
    
    for ext in extensions:
        for file_path in OUTPUTS_DIR.glob(f"*{ext}"):
            age_days = get_file_age_days(file_path)
            
            if age_days > retention_days:
                file_size = file_path.stat().st_size
                logger.info(f"  🗑️  {file_path.name} (age: {age_days} days, size: {file_size/1024:.2f} KB)")
                
                if not dry_run:
                    if secure_delete_file(file_path):
                        deleted_count += 1
                        freed_bytes += file_size
                else:
                    deleted_count += 1
                    freed_bytes += file_size
    
    logger.info(f"\nDeleted: {deleted_count} files")
    logger.info(f"Freed: {freed_bytes/1024/1024:.2f} MB")
    
    return deleted_count, freed_bytes

def cleanup_orphaned_files(dry_run=False):
    """
    Clean up files that have no corresponding audit log
    
    Args:
        dry_run: If True, don't actually delete files
        
    Returns:
        Tuple of (deleted_count, freed_bytes)
    """
    logger.info("\n" + "=" * 70)
    logger.info("CLEANING UP ORPHANED FILES")
    logger.info("=" * 70)
    
    if not OUTPUTS_DIR.exists() or not LOGS_DIR.exists():
        return 0, 0
    
    # Get all case IDs from logs
    case_ids = set()
    for log_file in LOGS_DIR.glob("*.json"):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                case_id = log_data.get('case_id', '')
                if case_id:
                    case_ids.add(case_id)
        except Exception as e:
            logger.warning(f"Could not read {log_file}: {e}")
    
    logger.info(f"Found {len(case_ids)} cases with logs")
    
    deleted_count = 0
    freed_bytes = 0
    
    # Check evidence files
    for file_path in OUTPUTS_DIR.glob("case_*"):
        # Extract case ID from filename
        filename = file_path.stem  # Remove extension
        parts = filename.split('_')
        
        if len(parts) >= 2:
            case_id = f"{parts[0]}_{parts[1]}"  # case_XXXX
            
            if case_id not in case_ids:
                file_size = file_path.stat().st_size
                logger.info(f"  🗑️  {file_path.name} (orphaned, no log found)")
                
                if not dry_run:
                    if secure_delete_file(file_path):
                        deleted_count += 1
                        freed_bytes += file_size
                else:
                    deleted_count += 1
                    freed_bytes += file_size
    
    logger.info(f"\nDeleted: {deleted_count} orphaned files")
    logger.info(f"Freed: {freed_bytes/1024:.2f} KB")
    
    return deleted_count, freed_bytes

def main():
    parser = argparse.ArgumentParser(description="AIShield Compliance Cleanup Script")
    parser.add_argument('--dry-run', action='store_true', 
                       help="Show what would be deleted without actually deleting")
    parser.add_argument('--force', action='store_true',
                       help="Skip confirmation prompts")
    parser.add_argument('--archive', action='store_true',
                       help="Archive logs before deletion")
    parser.add_argument('--orphaned-only', action='store_true',
                       help="Only clean up orphaned files")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("AISHIELD COMPLIANCE CLEANUP")
    logger.info("=" * 70)
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    logger.info(f"Archive: {'Yes' if args.archive else 'No'}")
    logger.info(f"Force: {'Yes' if args.force else 'No'}")
    logger.info("")
    
    # Check if cleanup is enabled
    if not COMPLIANCE_CONFIG.get("cleanup_enabled", True):
        logger.warning("⚠️  Cleanup is disabled in compliance configuration!")
        if not args.force:
            return
    
    # Confirmation
    if not args.dry_run and not args.force:
        response = input("\n⚠️  This will permanently delete old files. Continue? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Cleanup cancelled.")
            return
    
    # Perform cleanup
    total_deleted = 0
    total_archived = 0
    total_freed = 0
    
    if args.orphaned_only:
        # Only clean orphaned files
        deleted, freed = cleanup_orphaned_files(args.dry_run)
        total_deleted += deleted
        total_freed += freed
    else:
        # Full cleanup
        deleted_logs, archived, freed_logs = cleanup_old_logs(args.dry_run, args.archive)
        total_deleted += deleted_logs
        total_archived += archived
        total_freed += freed_logs
        
        deleted_evidence, freed_evidence = cleanup_old_evidence(args.dry_run)
        total_deleted += deleted_evidence
        total_freed += freed_evidence
        
        deleted_orphaned, freed_orphaned = cleanup_orphaned_files(args.dry_run)
        total_deleted += deleted_orphaned
        total_freed += freed_orphaned
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total files deleted: {total_deleted}")
    if total_archived > 0:
        logger.info(f"Total files archived: {total_archived}")
    logger.info(f"Total space freed: {total_freed/1024/1024:.2f} MB")
    
    if args.dry_run:
        logger.info("\n✅ DRY RUN COMPLETE - No files were actually deleted")
    else:
        logger.info("\n✅ CLEANUP COMPLETE")

if __name__ == "__main__":
    main()
