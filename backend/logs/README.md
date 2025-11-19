# Audit Logs Directory

This folder contains **JSON case logs** generated during runtime when users perform identity verification.

## Log File Format

Each verification creates a timestamped JSON file:
```
YYYY-MM-DD_HH-MM-SS_case_XXXX.json
```

## Log Contents

Each log file contains:
- **Case Information**: Case number, timestamp, session ID
- **Verification Results**: Document authenticity, liveness detection, face matching scores
- **Risk Assessment**: Fraud probability, confidence level, final verdict
- **SHAP Explanations**: Feature importance for model decisions
- **Compliance Data**: Processing metadata for audit trail

## Purpose

These logs enable:
- ✅ Regulatory compliance (KYC/AML)
- ✅ Audit trail for investigations
- ✅ Admin dashboard visualization
- ✅ Performance monitoring
- ✅ Forensic analysis

---

**Note**: This folder is automatically managed by the backend. Old logs are cleared during fresh deployments but preserved during runtime for admin review.
