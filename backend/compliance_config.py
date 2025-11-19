"""
Compliance Configuration for AIShield KYC System

This module defines security and compliance settings for:
- PII data handling
- File retention policies
- Data anonymization
- Secure deletion
- GDPR-like compliance principles

Author: AIShield Team
Date: November 15, 2025
"""

import os
from datetime import timedelta

# ============================================================================
# COMPLIANCE CONFIGURATION
# ============================================================================

COMPLIANCE_CONFIG = {
    # --- FILE DELETION POLICY ---
    "delete_raw_images": True,  # Delete original uploaded files after processing
    "delete_raw_videos": True,  # Delete original video files after processing
    "keep_processed_only": True,  # Keep only processed evidence (heatmaps, thumbnails)
    
    # --- DATA RETENTION POLICY ---
    "retain_days": 1,  # Number of days to retain logs and evidence
    "retention_period": timedelta(days=1),  # Timedelta for easy calculations
    
    # --- ANONYMIZATION ---
    "anonymize_filenames": True,  # Replace user filenames with anonymized names
    "anonymize_ip_in_logs": False,  # Hash/mask IP addresses in logs (set True for production)
    "redact_pii_from_logs": False,  # Remove PII fields from logs (set True if needed)
    
    # --- SECURITY SETTINGS ---
    "secure_file_deletion": True,  # Securely overwrite files before deletion
    "max_file_size_mb": 50,  # Maximum upload file size in MB
    "allowed_extensions": {
        "image": [".jpg", ".jpeg", ".png"],
        "video": [".mp4", ".mov", ".avi"]
    },
    
    # --- AUDIT TRAIL ---
    "log_retention_days": 90,  # Separate retention for audit logs (regulatory requirement)
    "archive_old_logs": True,  # Archive logs before deletion
    "archive_path": "logs/archive",  # Path for archived logs
    
    # --- GDPR COMPLIANCE ---
    "enable_data_export": True,  # Allow users to export their data
    "enable_right_to_delete": True,  # Allow users to request data deletion
    "consent_required": False,  # Require explicit consent (set True in production)
    
    # --- FILE NAMING PATTERNS ---
    "filename_patterns": {
        "id_document": "case_{case_id}_id{ext}",
        "selfie": "case_{case_id}_selfie{ext}",
        "video": "case_{case_id}_video{ext}",
        "heatmap": "case_{case_id}_heatmap.png",
        "thumbnail": "case_{case_id}_thumb.jpg"
    },
    
    # --- CLEANUP SCHEDULE ---
    "cleanup_enabled": True,  # Enable automatic cleanup
    "cleanup_hour": 2,  # Hour to run daily cleanup (2 AM)
    "cleanup_on_startup": False,  # Run cleanup when application starts
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(key: str, default=None):
    """Get a configuration value by key"""
    return COMPLIANCE_CONFIG.get(key, default)

def is_file_deletion_enabled() -> bool:
    """Check if raw file deletion is enabled"""
    return COMPLIANCE_CONFIG.get("delete_raw_images", False)

def is_anonymization_enabled() -> bool:
    """Check if filename anonymization is enabled"""
    return COMPLIANCE_CONFIG.get("anonymize_filenames", False)

def get_retention_days() -> int:
    """Get the number of days to retain data"""
    return COMPLIANCE_CONFIG.get("retain_days", 1)

def get_log_retention_days() -> int:
    """Get the number of days to retain audit logs"""
    return COMPLIANCE_CONFIG.get("log_retention_days", 90)

def generate_anonymous_filename(file_type: str, case_id: str, original_ext: str = ".jpg") -> str:
    """
    Generate an anonymized filename based on case ID and file type
    
    Args:
        file_type: Type of file ("id_document", "selfie", "video", "heatmap", "thumbnail")
        case_id: Case identifier
        original_ext: Original file extension
        
    Returns:
        Anonymized filename
    """
    patterns = COMPLIANCE_CONFIG["filename_patterns"]
    pattern = patterns.get(file_type, "case_{case_id}_file{ext}")
    
    # Extract extension if not provided
    if not original_ext.startswith("."):
        original_ext = f".{original_ext}"
    
    # Generate filename
    filename = pattern.format(case_id=case_id, ext=original_ext)
    return filename

def is_allowed_file(filename: str, file_category: str = "image") -> bool:
    """
    Check if file extension is allowed
    
    Args:
        filename: Original filename
        file_category: "image" or "video"
        
    Returns:
        True if extension is allowed
    """
    allowed = COMPLIANCE_CONFIG["allowed_extensions"].get(file_category, [])
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed

def get_max_file_size_bytes() -> int:
    """Get maximum file size in bytes"""
    return COMPLIANCE_CONFIG.get("max_file_size_mb", 50) * 1024 * 1024

# ============================================================================
# COMPLIANCE REPORT
# ============================================================================

def get_compliance_summary() -> dict:
    """
    Generate a compliance configuration summary
    
    Returns:
        Dictionary with compliance settings summary
    """
    return {
        "data_protection": {
            "raw_file_deletion": COMPLIANCE_CONFIG["delete_raw_images"],
            "filename_anonymization": COMPLIANCE_CONFIG["anonymize_filenames"],
            "secure_deletion": COMPLIANCE_CONFIG["secure_file_deletion"],
        },
        "retention_policy": {
            "evidence_retention_days": COMPLIANCE_CONFIG["retain_days"],
            "log_retention_days": COMPLIANCE_CONFIG["log_retention_days"],
            "archive_enabled": COMPLIANCE_CONFIG["archive_old_logs"],
        },
        "gdpr_compliance": {
            "data_export_enabled": COMPLIANCE_CONFIG["enable_data_export"],
            "right_to_delete_enabled": COMPLIANCE_CONFIG["enable_right_to_delete"],
            "consent_required": COMPLIANCE_CONFIG["consent_required"],
        },
        "security": {
            "max_file_size_mb": COMPLIANCE_CONFIG["max_file_size_mb"],
            "allowed_image_extensions": COMPLIANCE_CONFIG["allowed_extensions"]["image"],
            "allowed_video_extensions": COMPLIANCE_CONFIG["allowed_extensions"]["video"],
        }
    }

if __name__ == "__main__":
    print("=" * 70)
    print("COMPLIANCE CONFIGURATION SUMMARY")
    print("=" * 70)
    
    import json
    summary = get_compliance_summary()
    print(json.dumps(summary, indent=2))
    
    print("\n" + "=" * 70)
    print("FILENAME ANONYMIZATION EXAMPLES")
    print("=" * 70)
    
    case_id = "case_0001"
    print(f"ID Document:  {generate_anonymous_filename('id_document', case_id, '.jpg')}")
    print(f"Selfie:       {generate_anonymous_filename('selfie', case_id, '.jpg')}")
    print(f"Video:        {generate_anonymous_filename('video', case_id, '.mp4')}")
    print(f"Heatmap:      {generate_anonymous_filename('heatmap', case_id)}")
    print(f"Thumbnail:    {generate_anonymous_filename('thumbnail', case_id)}")
    
    print("\n" + "=" * 70)
    print("RETENTION POLICY")
    print("=" * 70)
    print(f"Evidence Retention: {get_retention_days()} days")
    print(f"Log Retention: {get_log_retention_days()} days")
    print(f"Cleanup Enabled: {COMPLIANCE_CONFIG['cleanup_enabled']}")
    print(f"Cleanup Hour: {COMPLIANCE_CONFIG['cleanup_hour']}:00")
