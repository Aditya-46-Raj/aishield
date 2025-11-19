"""
Audit Trail Logger for KYC Verification System
Logs every verification case with comprehensive metadata for compliance and governance
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory for logs
LOGS_DIR = Path(__file__).parent / "logs"

# Counter for case numbers (in production, use database sequence)
_case_counter = 0
_counter_lock_file = LOGS_DIR / ".case_counter"


def _get_next_case_number() -> int:
    """
    Get next case number with persistence
    In production, use database sequence or Redis counter
    """
    global _case_counter
    
    # Create logs dir if needed
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load counter from file if exists
    if _counter_lock_file.exists():
        try:
            with open(_counter_lock_file, 'r') as f:
                _case_counter = int(f.read().strip())
        except Exception as e:
            logger.warning(f"Could not read counter file: {e}")
            _case_counter = 0
    
    # Increment
    _case_counter += 1
    
    # Save counter
    try:
        with open(_counter_lock_file, 'w') as f:
            f.write(str(_case_counter))
    except Exception as e:
        logger.error(f"Could not save counter: {e}")
    
    return _case_counter


def log_verification(case_data: Dict[str, Any]) -> Optional[str]:
    """
    Log a KYC verification case to JSON file
    
    Args:
        case_data: Dictionary containing verification results and metadata
            Required fields:
            - doc_score: Document forgery score (0-1)
            - deepfake_score: Anti-spoof/deepfake detection score (0-1)
            - liveness_score: Overall liveness score (0-1)
            - embed_sim: Face embedding similarity (0-1)
            - fused_fraud_prob: Final fraud probability from fusion model (0-1)
            - risk_label: Risk classification (e.g., "CLEAN", "SUSPICIOUS", "FRAUD")
            - verdict: Overall verdict (e.g., "PASS", "FAIL", "REVIEW")
            
            Optional fields:
            - heatmap_path: Path to ELA heatmap image
            - model_versions: Dict of model names and versions
            - processing_time_ms: Total processing time in milliseconds
            - user_id: User identifier (if available)
            - session_id: Session identifier
            - ip_address: Client IP address
            - additional_metadata: Any extra metadata
    
    Returns:
        Path to saved log file, or None if failed
    """
    try:
        # Ensure logs directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp and case number
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        case_number = _get_next_case_number()
        case_id = f"case_{case_number:04d}"
        
        # Generate filename: YYYY-MM-DD_HH-MM-SS_case_XXXX.json
        filename = f"{timestamp_str}_{case_id}.json"
        log_path = LOGS_DIR / filename
        
        # Build comprehensive log structure
        log_entry = {
            # Case identification
            "case_id": case_id,
            "case_number": case_number,
            "timestamp": timestamp.isoformat(),
            "timestamp_unix": int(timestamp.timestamp()),
            
            # Verification scores
            "scores": {
                "document_score": case_data.get("doc_score"),
                "deepfake_score": case_data.get("deepfake_score"),
                "liveness_score": case_data.get("liveness_score"),
                "embedding_similarity": case_data.get("embed_sim"),
                "fused_fraud_probability": case_data.get("fused_fraud_prob"),
            },
            
            # Risk assessment
            "risk_assessment": {
                "risk_label": case_data.get("risk_label", "UNKNOWN"),
                "verdict": case_data.get("verdict", "PENDING"),
                "confidence": case_data.get("confidence"),
            },
            
            # Evidence and artifacts
            "evidence": {
                "heatmap_path": case_data.get("heatmap_path"),
                "document_path": case_data.get("document_path"),
                "video_path": case_data.get("video_path"),
                "selfie_path": case_data.get("selfie_path"),
            },
            
            # Model information
            "models": case_data.get("model_versions", {
                "document_classifier": "EfficientNet-B0 v1.0",
                "antispoof_detector": "TextureFrequency v1.0",
                "face_embedder": "InceptionResnetV1-VGGFace2",
                "fusion_model": "LightGBM v1.0"
            }),
            
            # Performance metrics
            "performance": {
                "processing_time_ms": case_data.get("processing_time_ms"),
                "frames_analyzed": case_data.get("frames_analyzed"),
                "detection_latency_ms": case_data.get("detection_latency_ms"),
            },
            
            # Session and user context
            "context": {
                "user_id": case_data.get("user_id"),
                "session_id": case_data.get("session_id"),
                "ip_address": case_data.get("ip_address"),
                "user_agent": case_data.get("user_agent"),
            },
            
            # Detailed component results
            "components": {
                "document_analysis": case_data.get("document_details", {}),
                "liveness_details": case_data.get("liveness_details", {}),
                "face_matching": case_data.get("face_matching_details", {}),
                "fusion_explanation": case_data.get("fusion_explanation", {}),
            },
            
            # Additional metadata
            "metadata": case_data.get("additional_metadata", {}),
            
            # System information
            "system": {
                "version": "AIShield v1.0",
                "environment": case_data.get("environment", "production"),
                "log_format_version": "1.0",
            }
        }
        
        # Write to JSON file with pretty formatting
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Audit log created: {filename}")
        logger.info(f"   Case ID: {case_id}")
        logger.info(f"   Verdict: {log_entry['risk_assessment']['verdict']}")
        logger.info(f"   Risk Label: {log_entry['risk_assessment']['risk_label']}")
        logger.info(f"   Fraud Probability: {log_entry['scores']['fused_fraud_probability']:.3f}")
        
        return str(log_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to create audit log: {e}")
        logger.exception(e)
        return None


def get_case_logs(limit: int = 10, filter_verdict: Optional[str] = None) -> list:
    """
    Retrieve recent case logs
    
    Args:
        limit: Maximum number of logs to return
        filter_verdict: Optional filter by verdict (e.g., "FAIL", "REVIEW")
    
    Returns:
        List of log entries (most recent first)
    """
    try:
        if not LOGS_DIR.exists():
            return []
        
        # Get all log files sorted by timestamp (newest first)
        log_files = sorted(LOGS_DIR.glob("*.json"), reverse=True)
        
        results = []
        for log_file in log_files[:limit * 2]:  # Read more in case of filtering
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_entry = json.load(f)
                
                # Apply filter if specified
                if filter_verdict:
                    if log_entry.get("risk_assessment", {}).get("verdict") != filter_verdict:
                        continue
                
                results.append(log_entry)
                
                if len(results) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Could not read log {log_file}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to retrieve case logs: {e}")
        return []


def get_case_by_id(case_id: str) -> Optional[Dict]:
    """
    Retrieve a specific case by case_id
    
    Args:
        case_id: Case identifier (e.g., "case_0001")
    
    Returns:
        Log entry dict or None if not found
    """
    try:
        if not LOGS_DIR.exists():
            return None
        
        # Search for file containing this case_id
        for log_file in LOGS_DIR.glob("*_" + case_id + ".json"):
            with open(log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to retrieve case {case_id}: {e}")
        return None


def get_log_statistics() -> Dict[str, Any]:
    """
    Get summary statistics from all logs
    
    Returns:
        Dictionary with statistics
    """
    try:
        if not LOGS_DIR.exists():
            return {"total_cases": 0}
        
        log_files = list(LOGS_DIR.glob("*.json"))
        
        stats = {
            "total_cases": len(log_files),
            "verdicts": {},
            "risk_labels": {},
            "average_fraud_prob": 0.0,
        }
        
        fraud_probs = []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_entry = json.load(f)
                
                # Count verdicts
                verdict = log_entry.get("risk_assessment", {}).get("verdict", "UNKNOWN")
                stats["verdicts"][verdict] = stats["verdicts"].get(verdict, 0) + 1
                
                # Count risk labels
                risk_label = log_entry.get("risk_assessment", {}).get("risk_label", "UNKNOWN")
                stats["risk_labels"][risk_label] = stats["risk_labels"].get(risk_label, 0) + 1
                
                # Collect fraud probabilities
                fraud_prob = log_entry.get("scores", {}).get("fused_fraud_probability")
                if fraud_prob is not None:
                    fraud_probs.append(fraud_prob)
                    
            except Exception as e:
                logger.warning(f"Could not read log {log_file}: {e}")
                continue
        
        if fraud_probs:
            stats["average_fraud_prob"] = sum(fraud_probs) / len(fraud_probs)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return {"total_cases": 0, "error": str(e)}


if __name__ == "__main__":
    # Test the audit logger
    print("\n" + "="*70)
    print("AUDIT LOGGER TEST")
    print("="*70)
    
    # Create sample case data
    sample_case = {
        "doc_score": 0.234,
        "deepfake_score": 0.132,
        "liveness_score": 0.654,
        "embed_sim": 0.892,
        "fused_fraud_prob": 0.156,
        "risk_label": "CLEAN",
        "verdict": "PASS",
        "confidence": 0.95,
        "heatmap_path": "outputs/sample_ela_heatmap.png",
        "document_path": "samples/clean_id.jpg",
        "video_path": "samples/video_sample.mp4",
        "processing_time_ms": 1234.56,
        "frames_analyzed": 10,
        "user_id": "user_12345",
        "session_id": "session_abc123",
        "ip_address": "192.168.1.100",
    }
    
    # Test logging
    print("\n📝 Creating audit log...")
    log_path = log_verification(sample_case)
    
    if log_path:
        print(f"\n✅ Log created successfully: {log_path}")
        
        # Read and display the log
        with open(log_path, 'r') as f:
            log_content = json.load(f)
        
        print("\n📄 Log Content Preview:")
        print(json.dumps(log_content, indent=2)[:1000] + "...")
        
        # Test retrieval
        print("\n📊 Testing log retrieval...")
        recent_logs = get_case_logs(limit=5)
        print(f"   Retrieved {len(recent_logs)} recent logs")
        
        # Test statistics
        print("\n📈 Testing statistics...")
        stats = get_log_statistics()
        print(f"   Total cases: {stats['total_cases']}")
        print(f"   Verdicts: {stats['verdicts']}")
        print(f"   Average fraud probability: {stats['average_fraud_prob']:.3f}")
    else:
        print("\n❌ Failed to create log")
    
    print("\n" + "="*70)
