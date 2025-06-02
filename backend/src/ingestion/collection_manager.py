"""
Collection Manager for organizing documents into specialized collections.
Handles automatic document classification and collection assignment.
"""

import re
from typing import Dict, List, Optional, Set
from pathlib import Path
from loguru import logger

from backend.src.models.data_models import DocumentCollection

class CollectionManager:
    """Manages document collection assignment and organization."""
    
    def __init__(self):
        self.collection_rules = self._initialize_collection_rules()
        
    def _initialize_collection_rules(self) -> Dict[DocumentCollection, Dict[str, List[str]]]:
        """Initialize rules for automatic collection assignment."""
        return {
            DocumentCollection.SOP_POLICY: {
                "filename_patterns": [
                    r".*sop.*",
                    r".*policy.*",
                    r".*procedure.*",
                    r".*protocol.*",
                    r".*standard.*operating.*"
                ],
                "content_patterns": [
                    r"standard operating procedure",
                    r"company policy",
                    r"effective date",
                    r"revision history",
                    r"approval.*signature"
                ]
            },
            DocumentCollection.REGULATORY: {
                "filename_patterns": [
                    r".*fda.*",
                    r".*regulatory.*",
                    r".*compliance.*",
                    r".*approval.*",
                    r".*submission.*"
                ],
                "content_patterns": [
                    r"fda.*approval",
                    r"regulatory.*requirement",
                    r"compliance.*standard",
                    r"510\(k\)",
                    r"medical device"
                ]
            },
            DocumentCollection.RESEARCH: {
                "filename_patterns": [
                    r".*research.*",
                    r".*study.*",
                    r".*analysis.*",
                    r".*experiment.*",
                    r".*findings.*"
                ],
                "content_patterns": [
                    r"methodology",
                    r"statistical.*analysis",
                    r"hypothesis",
                    r"correlation",
                    r"p.*value"
                ]
            },
            DocumentCollection.CLINICAL: {
                "filename_patterns": [
                    r".*clinical.*",
                    r".*trial.*",
                    r".*patient.*",
                    r".*diagnostic.*"
                ],
                "content_patterns": [
                    r"clinical.*trial",
                    r"patient.*data",
                    r"diagnostic.*accuracy",
                    r"breath.*sample",
                    r"biomarker"
                ]
            },
            DocumentCollection.LEGACY: {
                "filename_patterns": [
                    r".*legacy.*",
                    r".*archive.*",
                    r".*old.*",
                    r".*deprecated.*",
                    r".*\d{4}.*"  # Year in filename
                ],
                "content_patterns": [
                    r"deprecated",
                    r"superseded",
                    r"archived",
                    r"no longer.*valid"
                ]
            },
            DocumentCollection.TRAINING: {
                "filename_patterns": [
                    r".*training.*",
                    r".*manual.*",
                    r".*guide.*",
                    r".*tutorial.*",
                    r".*instruction.*"
                ],
                "content_patterns": [
                    r"training.*material",
                    r"user.*guide",
                    r"step.*by.*step",
                    r"how.*to",
                    r"instruction.*manual"
                ]
            }
        }
    
    def classify_document(
        self, 
        file_path: str, 
        content_sample: str = "",
        explicit_collection: Optional[DocumentCollection] = None
    ) -> DocumentCollection:
        """
        Classify a document into the appropriate collection.
        
        Args:
            file_path: Path to the document
            content_sample: Sample of document content for analysis
            explicit_collection: Override automatic classification
            
        Returns:
            DocumentCollection enum value
        """
        if explicit_collection:
            logger.info(f"Using explicit collection assignment: {explicit_collection}")
            return explicit_collection
            
        filename = Path(file_path).name.lower()
        content_lower = content_sample.lower()
        
        # Calculate confidence scores for each collection
        scores = {}
        
        for collection, rules in self.collection_rules.items():
            score = 0
            
            # Check filename patterns
            for pattern in rules.get("filename_patterns", []):
                if re.search(pattern, filename, re.IGNORECASE):
                    score += 2
                    
            # Check content patterns (if content available)
            if content_sample:
                for pattern in rules.get("content_patterns", []):
                    matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                    score += matches
                    
            scores[collection] = score
        
        # Get collection with highest score
        if scores and max(scores.values()) > 0:
            best_collection = max(scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Auto-classified '{filename}' as {best_collection} (score: {scores[best_collection]})")
            return best_collection
        
        # Default to current documents if no match
        logger.info(f"No classification match for '{filename}', defaulting to CURRENT")
        return DocumentCollection.CURRENT
    
    def get_collection_info(self) -> Dict[DocumentCollection, Dict[str, str]]:
        """Get display information for all collections."""
        return {
            DocumentCollection.CURRENT: {
                "display_name": "Current Documents",
                "description": "Active documents and latest implementations"
            },
            DocumentCollection.LEGACY: {
                "display_name": "Legacy/Historical",
                "description": "Archived and historical documents"
            },
            DocumentCollection.SOP_POLICY: {
                "display_name": "SOPs & Policies", 
                "description": "Standard Operating Procedures and company policies"
            },
            DocumentCollection.RESEARCH: {
                "display_name": "Research Data",
                "description": "Research findings, studies, and analytical data"
            },
            DocumentCollection.CLINICAL: {
                "display_name": "Clinical Studies",
                "description": "Clinical trial data and diagnostic studies"
            },
            DocumentCollection.REGULATORY: {
                "display_name": "Regulatory Documents",
                "description": "FDA submissions, compliance documents, and approvals"
            },
            DocumentCollection.TRAINING: {
                "display_name": "Training Materials", 
                "description": "User guides, manuals, and training resources"
            }
        }
    
    def validate_collection(self, collection_name: str) -> Optional[DocumentCollection]:
        """Validate and convert string to DocumentCollection enum."""
        try:
            return DocumentCollection(collection_name)
        except ValueError:
            logger.warning(f"Invalid collection name: {collection_name}")
            return None 