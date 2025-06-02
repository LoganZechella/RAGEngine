"""
Collection Manager for Breath Diagnostics RAGEngine.
Handles automatic document classification and collection management.
"""

import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from ..models.data_models import DocumentCollection, CollectionMetadata


class CollectionManager:
    """Manages document collections with automatic classification."""
    
    def __init__(self):
        self.collection_metadata = self._initialize_collection_metadata()
        self.classification_rules = self._initialize_classification_rules()
    
    def _initialize_collection_metadata(self) -> Dict[DocumentCollection, CollectionMetadata]:
        """Initialize metadata for all collections."""
        return {
            DocumentCollection.CURRENT: CollectionMetadata(
                collection_name=DocumentCollection.CURRENT,
                display_name="Current Documents",
                description="Active operational documents and recent publications",
                auto_assignment_rules=[
                    "filename contains: current, active, latest, 2024, 2025",
                    "content contains: current protocol, active procedure"
                ]
            ),
            DocumentCollection.LEGACY: CollectionMetadata(
                collection_name=DocumentCollection.LEGACY,
                display_name="Legacy Documents",
                description="Historical documents and archived materials",
                auto_assignment_rules=[
                    "filename contains: legacy, archive, old, historical, 2020, 2021, 2022, 2023",
                    "content contains: archived, superseded, historical"
                ]
            ),
            DocumentCollection.SOP_POLICY: CollectionMetadata(
                collection_name=DocumentCollection.SOP_POLICY,
                display_name="SOPs & Policies",
                description="Standard Operating Procedures and organizational policies",
                auto_assignment_rules=[
                    "filename contains: sop, policy, procedure, protocol, guideline",
                    "content contains: standard operating procedure, policy, protocol"
                ]
            ),
            DocumentCollection.RESEARCH: CollectionMetadata(
                collection_name=DocumentCollection.RESEARCH,
                display_name="Research Data",
                description="Research studies, experimental data, and scientific publications",
                auto_assignment_rules=[
                    "filename contains: research, study, experiment, data, analysis",
                    "content contains: research, study, experiment, hypothesis, methodology"
                ]
            ),
            DocumentCollection.CLINICAL: CollectionMetadata(
                collection_name=DocumentCollection.CLINICAL,
                display_name="Clinical Studies",
                description="Clinical trials, patient studies, and medical research",
                auto_assignment_rules=[
                    "filename contains: clinical, trial, patient, medical, diagnostic",
                    "content contains: clinical trial, patient, diagnosis, treatment, medical"
                ]
            ),
            DocumentCollection.REGULATORY: CollectionMetadata(
                collection_name=DocumentCollection.REGULATORY,
                display_name="Regulatory Documents",
                description="Regulatory compliance, FDA submissions, and legal documents",
                auto_assignment_rules=[
                    "filename contains: regulatory, fda, compliance, legal, regulation",
                    "content contains: FDA, regulatory, compliance, regulation, legal"
                ]
            ),
            DocumentCollection.TRAINING: CollectionMetadata(
                collection_name=DocumentCollection.TRAINING,
                display_name="Training Materials",
                description="Training manuals, educational content, and learning resources",
                auto_assignment_rules=[
                    "filename contains: training, manual, guide, tutorial, education",
                    "content contains: training, tutorial, guide, instruction, learning"
                ]
            )
        }
    
    def _initialize_classification_rules(self) -> Dict[DocumentCollection, Dict[str, List[str]]]:
        """Initialize classification rules for automatic document assignment."""
        return {
            DocumentCollection.SOP_POLICY: {
                "filename_patterns": [
                    r"sop[_\-\s]",
                    r"policy[_\-\s]",
                    r"procedure[_\-\s]",
                    r"protocol[_\-\s]",
                    r"guideline[_\-\s]",
                    r"standard[_\-\s]operating",
                ],
                "content_patterns": [
                    r"standard\s+operating\s+procedure",
                    r"policy\s+number",
                    r"procedure\s+for",
                    r"protocol\s+version",
                    r"guideline\s+for",
                ]
            },
            DocumentCollection.CLINICAL: {
                "filename_patterns": [
                    r"clinical[_\-\s]",
                    r"trial[_\-\s]",
                    r"patient[_\-\s]",
                    r"diagnostic[_\-\s]",
                    r"breath[_\-\s]test",
                    r"medical[_\-\s]",
                ],
                "content_patterns": [
                    r"clinical\s+trial",
                    r"patient\s+study",
                    r"breath\s+analysis",
                    r"diagnostic\s+accuracy",
                    r"medical\s+device",
                    r"biomarker\s+detection",
                ]
            },
            DocumentCollection.RESEARCH: {
                "filename_patterns": [
                    r"research[_\-\s]",
                    r"study[_\-\s]",
                    r"experiment[_\-\s]",
                    r"analysis[_\-\s]",
                    r"data[_\-\s]",
                    r"publication[_\-\s]",
                ],
                "content_patterns": [
                    r"research\s+methodology",
                    r"experimental\s+design",
                    r"data\s+analysis",
                    r"statistical\s+significance",
                    r"hypothesis\s+testing",
                    r"peer\s+review",
                ]
            },
            DocumentCollection.REGULATORY: {
                "filename_patterns": [
                    r"fda[_\-\s]",
                    r"regulatory[_\-\s]",
                    r"compliance[_\-\s]",
                    r"regulation[_\-\s]",
                    r"legal[_\-\s]",
                    r"submission[_\-\s]",
                ],
                "content_patterns": [
                    r"fda\s+approval",
                    r"regulatory\s+compliance",
                    r"cfr\s+\d+",
                    r"iso\s+\d+",
                    r"quality\s+management\s+system",
                    r"medical\s+device\s+regulation",
                ]
            },
            DocumentCollection.TRAINING: {
                "filename_patterns": [
                    r"training[_\-\s]",
                    r"manual[_\-\s]",
                    r"guide[_\-\s]",
                    r"tutorial[_\-\s]",
                    r"instruction[_\-\s]",
                    r"education[_\-\s]",
                ],
                "content_patterns": [
                    r"training\s+manual",
                    r"user\s+guide",
                    r"step\s+by\s+step",
                    r"how\s+to\s+use",
                    r"operating\s+instructions",
                    r"learning\s+objectives",
                ]
            },
            DocumentCollection.LEGACY: {
                "filename_patterns": [
                    r"legacy[_\-\s]",
                    r"archive[_\-\s]",
                    r"old[_\-\s]",
                    r"historical[_\-\s]",
                    r"20(20|21|22|23)[_\-\s]",
                    r"v[01]\.",
                ],
                "content_patterns": [
                    r"archived\s+document",
                    r"superseded\s+by",
                    r"historical\s+reference",
                    r"no\s+longer\s+active",
                    r"replaced\s+by",
                ]
            },
            DocumentCollection.CURRENT: {
                "filename_patterns": [
                    r"current[_\-\s]",
                    r"active[_\-\s]",
                    r"latest[_\-\s]",
                    r"20(24|25)[_\-\s]",
                    r"v[2-9]\.",
                ],
                "content_patterns": [
                    r"current\s+version",
                    r"active\s+document",
                    r"latest\s+revision",
                    r"effective\s+date",
                    r"current\s+protocol",
                ]
            }
        }
    
    def classify_document(self, filename: str, content: str = "") -> DocumentCollection:
        """
        Automatically classify a document based on filename and content.
        
        Args:
            filename: The document filename
            content: The document content (optional)
            
        Returns:
            The most appropriate DocumentCollection
        """
        filename_lower = filename.lower()
        content_lower = content.lower() if content else ""
        
        # Score each collection based on pattern matches
        collection_scores = {}
        
        for collection, rules in self.classification_rules.items():
            score = 0
            
            # Check filename patterns
            for pattern in rules.get("filename_patterns", []):
                if re.search(pattern, filename_lower, re.IGNORECASE):
                    score += 2  # Filename matches are weighted higher
            
            # Check content patterns (if content is provided)
            if content:
                for pattern in rules.get("content_patterns", []):
                    matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                    score += matches  # Each content match adds 1 point
            
            collection_scores[collection] = score
        
        # Return the collection with the highest score
        if collection_scores:
            best_collection = max(collection_scores, key=collection_scores.get)
            if collection_scores[best_collection] > 0:
                return best_collection
        
        # Default to CURRENT if no patterns match
        return DocumentCollection.CURRENT
    
    def get_collection_metadata(self, collection: DocumentCollection) -> CollectionMetadata:
        """Get metadata for a specific collection."""
        return self.collection_metadata[collection]
    
    def get_all_collections_metadata(self) -> Dict[DocumentCollection, CollectionMetadata]:
        """Get metadata for all collections."""
        return self.collection_metadata.copy()
    
    def get_collection_display_info(self, collection: DocumentCollection) -> Dict[str, str]:
        """Get display information for a collection."""
        metadata = self.collection_metadata[collection]
        
        # Collection-specific icons and colors
        display_info = {
            DocumentCollection.CURRENT: {"icon": "📄", "color": "blue"},
            DocumentCollection.LEGACY: {"icon": "📜", "color": "gray"},
            DocumentCollection.SOP_POLICY: {"icon": "📋", "color": "green"},
            DocumentCollection.RESEARCH: {"icon": "🔬", "color": "purple"},
            DocumentCollection.CLINICAL: {"icon": "🏥", "color": "red"},
            DocumentCollection.REGULATORY: {"icon": "⚖️", "color": "yellow"},
            DocumentCollection.TRAINING: {"icon": "🎓", "color": "indigo"},
        }
        
        info = display_info.get(collection, {"icon": "📁", "color": "gray"})
        info.update({
            "name": metadata.display_name,
            "description": metadata.description,
            "collection_id": collection.value
        })
        
        return info
    
    def validate_collection_name(self, collection_name: str) -> Optional[DocumentCollection]:
        """Validate and convert a collection name string to DocumentCollection enum."""
        try:
            return DocumentCollection(collection_name)
        except ValueError:
            return None
    
    def get_breath_diagnostics_specific_rules(self) -> Dict[str, List[str]]:
        """Get breath diagnostics specific classification patterns."""
        return {
            "breath_analysis_keywords": [
                "breath analysis", "volatile organic compounds", "voc detection",
                "breath biomarkers", "exhaled breath", "breath sampling",
                "breath collection", "breath sensor", "breath diagnostic"
            ],
            "medical_device_keywords": [
                "medical device", "diagnostic device", "breath analyzer",
                "sensor technology", "detection system", "measurement device"
            ],
            "clinical_keywords": [
                "clinical validation", "patient study", "diagnostic accuracy",
                "sensitivity", "specificity", "clinical trial", "biomarker"
            ],
            "regulatory_keywords": [
                "fda clearance", "ce marking", "iso 13485", "medical device regulation",
                "clinical evaluation", "risk management", "quality system"
            ]
        }
    
    def suggest_collection_for_query(self, query: str) -> List[DocumentCollection]:
        """Suggest relevant collections based on a search query."""
        query_lower = query.lower()
        suggestions = []
        
        # Breath diagnostics specific suggestions
        breath_keywords = self.get_breath_diagnostics_specific_rules()
        
        # Check for clinical/medical terms
        if any(keyword in query_lower for keyword in breath_keywords["clinical_keywords"]):
            suggestions.append(DocumentCollection.CLINICAL)
        
        # Check for regulatory terms
        if any(keyword in query_lower for keyword in breath_keywords["regulatory_keywords"]):
            suggestions.append(DocumentCollection.REGULATORY)
        
        # Check for research terms
        if any(term in query_lower for term in ["research", "study", "analysis", "experiment"]):
            suggestions.append(DocumentCollection.RESEARCH)
        
        # Check for SOP/policy terms
        if any(term in query_lower for term in ["procedure", "protocol", "policy", "guideline"]):
            suggestions.append(DocumentCollection.SOP_POLICY)
        
        # Check for training terms
        if any(term in query_lower for term in ["training", "manual", "guide", "instruction"]):
            suggestions.append(DocumentCollection.TRAINING)
        
        # If no specific suggestions, include current and research as defaults
        if not suggestions:
            suggestions = [DocumentCollection.CURRENT, DocumentCollection.RESEARCH]
        
        return suggestions