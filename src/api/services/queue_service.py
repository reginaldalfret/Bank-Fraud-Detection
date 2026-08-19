"""Investigation Queue Service for Fraud Triage and Case Management.

Maintains fraud investigation queue state, supports analyst triage actions
(Review, Escalate, Mark Legitimate, Confirm Fraud, Add Notes), pre-populates
benchmark cases, and generates CSV queue exports.
"""

from __future__ import annotations

import csv
import io
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.api.schemas import NoteEntry, QueueActionRequest, QueueItem, QueueListResponse
from src.api.services.data_service import get_data_service
from src.api.services.model_service import get_model_service
from src.api.services.threshold_service import get_threshold_service

logger = logging.getLogger("fraud_api.queue_service")


class QueueService:
    """Enterprise Investigation Queue Management Engine."""

    def __init__(self):
        self._lock = threading.Lock()
        self.queue_items: Dict[str, QueueItem] = {}
        self._seed_sample_queue()

    def _seed_sample_queue(self):
        """Pre-populate investigation queue with flagged cases from data service."""
        data_svc = get_data_service()
        model_svc = get_model_service()
        thresh_svc = get_threshold_service()

        logger.info("Initializing triage queue from existing applications...")
        apps, _ = data_svc.list_applications(page=1, page_size=60)
        
        for app_dict in apps:
            from src.api.schemas import ApplicationRequest
            try:
                app_req = ApplicationRequest(**app_dict)
                pred = model_svc.predict_application(app_req)
                prob = pred["fraud_probability"]
                action, risk_level, _, _ = thresh_svc.evaluate_decision(prob, "balanced")
                
                # Enqueue items flagged for review or block
                if action in ("REVIEW", "BLOCK") or prob >= 0.015:
                    app_id = app_req.application_id
                    now_str = datetime.now(timezone.utc).isoformat()
                    
                    # Create initial notes
                    init_notes = [
                        NoteEntry(
                            timestamp=now_str,
                            analyst_id="SYSTEM",
                            action="FLAGGED",
                            note=f"Flagged by real-time ensemble model with fraud score {prob:.2%} ({risk_level})"
                        )
                    ]
                    
                    tags = ["auto_flagged"]
                    if action == "BLOCK":
                        tags.append("critical_risk")
                    if app_req.name_email_similarity < 0.2:
                        tags.append("synthetic_email")
                    if app_req.prev_address_months_count < 0:
                        tags.append("thin_file")

                    item = QueueItem(
                        application_id=app_id,
                        fraud_probability=prob,
                        risk_level=risk_level,
                        status="PENDING",
                        decision=action,
                        assigned_to=None,
                        notes_history=init_notes,
                        tags=tags,
                        created_at=now_str,
                        updated_at=now_str,
                        application_data=app_dict
                    )
                    self.queue_items[app_id] = item
            except Exception as ex:
                logger.debug("Skipped queuing application: %s", ex)

        logger.info("Triage queue initialized with %d active investigation cases", len(self.queue_items))

    def enqueue_application(
        self,
        application_id: str,
        fraud_probability: float,
        risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        decision: Literal["APPROVE", "REVIEW", "BLOCK"],
        application_data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> QueueItem:
        """Add or update an application in the active investigation queue."""
        with self._lock:
            now_str = datetime.now(timezone.utc).isoformat()
            existing = self.queue_items.get(application_id)
            
            if existing:
                existing.fraud_probability = fraud_probability
                existing.risk_level = risk_level
                existing.decision = decision
                existing.updated_at = now_str
                if tags:
                    existing.tags = list(set(existing.tags + tags))
                return existing

            init_tags = tags or ["realtime_alert"]
            if decision == "BLOCK":
                init_tags.append("high_priority")

            item = QueueItem(
                application_id=application_id,
                fraud_probability=fraud_probability,
                risk_level=risk_level,
                status="PENDING",
                decision=decision,
                assigned_to=None,
                notes_history=[
                    NoteEntry(
                        timestamp=now_str,
                        analyst_id="SYSTEM",
                        action="ENQUEUED",
                        note=f"Application routed to queue with score {fraud_probability:.2%} ({decision})"
                    )
                ],
                tags=init_tags,
                created_at=now_str,
                updated_at=now_str,
                application_data=application_data
            )
            self.queue_items[application_id] = item
            return item

    def perform_action(self, req: QueueActionRequest) -> QueueItem:
        """Apply an investigator triage action to a queue item."""
        with self._lock:
            item = self.queue_items.get(req.application_id)
            if not item:
                # If item was not in queue, fetch application and create it
                data_svc = get_data_service()
                raw_app = data_svc.get_application(req.application_id)
                now_str = datetime.now(timezone.utc).isoformat()
                item = QueueItem(
                    application_id=req.application_id,
                    fraud_probability=0.05,
                    risk_level="HIGH",
                    status="PENDING",
                    decision="REVIEW",
                    assigned_to=req.analyst_id,
                    notes_history=[],
                    tags=req.tags or ["manual_triage"],
                    created_at=now_str,
                    updated_at=now_str,
                    application_data=raw_app
                )
                self.queue_items[req.application_id] = item

            now_str = datetime.now(timezone.utc).isoformat()
            item.updated_at = now_str
            
            # Action Mapping
            action_key = req.action.strip()
            if action_key == "Review":
                item.status = "UNDER_REVIEW"
                item.assigned_to = req.analyst_id
                note_text = req.notes or f"Assigned to analyst {req.analyst_id} for KYC document verification."
            elif action_key == "Escalate":
                item.status = "ESCALATED"
                item.assigned_to = req.analyst_id
                if "escalated" not in item.tags:
                    item.tags.append("escalated")
                note_text = req.notes or f"Escalated to Senior Fraud Operations by {req.analyst_id}."
            elif action_key == "Mark Legitimate":
                item.status = "RESOLVED_LEGITIMATE"
                item.decision = "APPROVE"
                note_text = req.notes or f"Cleared as genuine customer by {req.analyst_id}."
            elif action_key == "Confirm Fraud":
                item.status = "RESOLVED_FRAUD"
                item.decision = "BLOCK"
                if "confirmed_fraud" not in item.tags:
                    item.tags.append("confirmed_fraud")
                note_text = req.notes or f"Confirmed account opening fraud by {req.analyst_id}. SAR filing initiated."
            elif action_key == "Add Notes":
                note_text = req.notes or "Note appended by analyst."
            else:
                note_text = req.notes or f"Action {action_key} performed."

            if req.tags:
                item.tags = list(set(item.tags + req.tags))

            item.notes_history.append(NoteEntry(
                timestamp=now_str,
                analyst_id=req.analyst_id,
                action=action_key,
                note=note_text
            ))

            return item

    def get_queue(
        self,
        status: Optional[str] = None,
        risk_level: Optional[str] = None,
        assigned_to: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> QueueListResponse:
        """Get filtered and paginated list of triage items."""
        with self._lock:
            items = list(self.queue_items.values())

        # Compute status counts across entire queue
        pending_cnt = sum(1 for x in items if x.status == "PENDING")
        review_cnt = sum(1 for x in items if x.status == "UNDER_REVIEW")
        escalated_cnt = sum(1 for x in items if x.status == "ESCALATED")
        resolved_cnt = sum(1 for x in items if x.status in ("RESOLVED_LEGITIMATE", "RESOLVED_FRAUD"))

        # Apply filters
        if status:
            items = [x for x in items if x.status.upper() == status.upper()]
        if risk_level:
            items = [x for x in items if x.risk_level.upper() == risk_level.upper()]
        if assigned_to:
            items = [x for x in items if x.assigned_to and x.assigned_to.lower() == assigned_to.lower()]

        # Sort: Escalate & Critical first, then by highest fraud score
        priority_order = {"ESCALATED": 0, "PENDING": 1, "UNDER_REVIEW": 2, "RESOLVED_FRAUD": 3, "RESOLVED_LEGITIMATE": 4}
        items.sort(key=lambda x: (priority_order.get(x.status, 5), -x.fraud_probability))

        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        paged_items = items[start:end]

        return QueueListResponse(
            total=total,
            pending_count=pending_cnt,
            under_review_count=review_cnt,
            escalated_count=escalated_cnt,
            resolved_count=resolved_cnt,
            items=paged_items
        )

    def export_queue_csv(self) -> str:
        """Export all investigation queue items to formatted CSV string."""
        with self._lock:
            items = list(self.queue_items.values())

        output = io.StringIO()
        writer = csv.writer(output)
        
        # CSV Headers
        writer.writerow([
            "application_id",
            "fraud_probability",
            "risk_level",
            "decision",
            "status",
            "assigned_to",
            "tags",
            "created_at",
            "updated_at",
            "latest_note"
        ])

        for item in items:
            latest_note = item.notes_history[-1].note if item.notes_history else ""
            writer.writerow([
                item.application_id,
                f"{item.fraud_probability:.4f}",
                item.risk_level,
                item.decision,
                item.status,
                item.assigned_to or "Unassigned",
                ";".join(item.tags),
                item.created_at,
                item.updated_at,
                latest_note.replace("\n", " ")
            ])

        return output.getvalue()


_queue_service_instance: Optional[QueueService] = None


def get_queue_service() -> QueueService:
    global _queue_service_instance
    if _queue_service_instance is None:
        _queue_service_instance = QueueService()
    return _queue_service_instance
