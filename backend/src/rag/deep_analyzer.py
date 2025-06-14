"""
Enhanced Deep Analyzer for RAG Engine.
Performs PhD-level comprehensive analysis with improved output structure and readability.
Optimized for the enhanced SynthesizedKnowledge data model.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
import json
import google.genai as genai
from google.genai import types
from datetime import datetime

from backend.src.models.data_models import (
    RetrievedContext, SynthesizedKnowledge, KeyConcept, SynthesisInsight, 
    ResearchGap, AnalysisDepth, EvidenceQuality
)


class DeepAnalyzer:
    """
    Enhanced Deep Analyzer for PhD-level comprehensive analysis with improved structure.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro-preview-06-05",
        temperature: float = 0.2,
        max_output_tokens: int = 20000,
        verbose: bool = False
    ):
        """Initialize the Enhanced DeepAnalyzer."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set it via parameter or GOOGLE_API_KEY environment variable.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.verbose = verbose
        
        self.client = genai.Client(api_key=self.api_key)
        
        system_instruction = """
Adopt the persona of a PhD-level researcher and critical analyst. 
Your task is to perform a comprehensive analysis of the provided source material.
"""
        
        self.config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            system_instruction=system_instruction
        )

    def synthesize_knowledge(
        self,
        query_details: Dict[str, Any],
        contexts: List[RetrievedContext]
    ) -> Optional[SynthesizedKnowledge]:
        """
        Perform enhanced PhD-level knowledge synthesis with structured output.
        """
        if not contexts:
            logger.warning("No contexts provided for knowledge synthesis")
            return SynthesizedKnowledge(
                summary="No context available for synthesis.",
                source_chunk_ids=[],
                analysis_depth=AnalysisDepth.ERROR,
                num_source_contexts=0
            )
        
        logger.info(f"Synthesizing knowledge from {len(contexts)} contexts")
        
        context_texts = [f"Context {i+1}:\n{ctx.text}\n" for i, ctx in enumerate(contexts)]
        consolidated_context = "\n\n".join(context_texts)
        
        prompt = self._create_enhanced_synthesis_prompt(query_details, consolidated_context)
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.config,
            )
            
            thoughts = []
            final_response_text = ""

            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text'):
                    final_response_text = part.text
                if hasattr(part, 'thought'):
                    thoughts.append(part.thought)

            if self.verbose:
                logger.debug(f"Gemini thoughts: {thoughts}")
                logger.debug(f"Gemini final response: {final_response_text}")

            synthesized = self._parse_enhanced_response(final_response_text, contexts, thoughts)
            
            if synthesized:
                logger.info(f"Successfully synthesized knowledge with {len(synthesized.key_concepts)} concepts, {len(synthesized.synthesis_insights)} insights")
            
            return synthesized
            
        except Exception as e:
            logger.error(f"Error in knowledge synthesis: {str(e)}")
            return SynthesizedKnowledge(
                summary=f"Error during synthesis: {str(e)}",
                source_chunk_ids=[ctx.chunk_id for ctx in contexts],
                analysis_depth=AnalysisDepth.ERROR,
                num_source_contexts=len(contexts)
            )
    
    def _create_enhanced_synthesis_prompt(self, query_details: Dict[str, Any], consolidated_context: str) -> str:
        """Create an enhanced prompt for structured knowledge synthesis."""
        
        domain = query_details.get("domain", "")
        task_description = query_details.get("task", "Analyze and synthesize information")
        query = query_details.get("query", "")
        
        prompt = f"""
I. Analysis Directive

Perform comprehensive analysis of the provided information related to: {query}
{f"Domain: {domain}" if domain else ""}

II. Source Material

{consolidated_context}

III. Required JSON Output Structure

Provide a comprehensive analysis in the following JSON format:

{{
  "summary": "<Comprehensive critical summary with analytical insights (2-3 paragraphs)>",
  "key_concepts": [
    {{
      "concept": "<Concept name>",
      "explanation": "<Clear, precise explanation>",
      "importance": "<Why this concept is important and its implications>",
      "evidence_quality": "<strong|moderate|weak|insufficient|conflicting>",
      "controversies": "<Any debates, limitations, or alternative perspectives>",
      "related_concepts": ["<Related concept 1>", "<Related concept 2>"],
      "confidence_score": <0.0-1.0>
    }}
  ],
  "topics": [
    "<Important topic or research area>",
    "<Another relevant topic>"
  ],
  "synthesis_insights": [
    {{
      "insight": "<Novel insight or connection discovered>",
      "supporting_evidence": ["<Evidence point 1>", "<Evidence point 2>"],
      "confidence_level": "<high|moderate|low>",
      "implications": "<Broader implications of this insight>"
    }}
  ],
  "research_gaps": [
    {{
      "gap_description": "<Description of identified gap or limitation>",
      "severity": "<critical|moderate|minor>",
      "suggested_investigation": "<Recommendations for further research>"
    }}
  ],
  "methodological_observations": "<Critical assessment of methodological approaches evident in the content>",
  "theoretical_implications": "<Broader theoretical or practical implications that emerge>",
  "overall_confidence": <0.0-1.0>,
  "completeness_score": <0.0-1.0>,
  "query_complexity": "<simple|moderate|complex|highly_complex>",
  "synthesis_quality_indicators": {{
    "cross_referencing_quality": "<high|moderate|low>",
    "analytical_depth": "<superficial|moderate|deep|comprehensive>",
    "novel_insights_generated": "<none|few|several|many>",
    "evidence_integration": "<poor|fair|good|excellent>"
  }}
}}

IV. Analysis Requirements

- Provide direct, actionable answers to the specific query
- Perform critical analysis beyond mere summarization
- Identify patterns, connections, and novel insights
- Assess evidence quality and highlight any controversies
- Generate synthesis insights that aren't explicitly stated in sources
- Identify research gaps and methodological considerations
- Assign realistic confidence scores based on evidence strength
- Ensure all claims are supported by the provided contexts

V. Reasoning Process

- Before generating the final JSON, use your thinking process to outline your analysis step-by-step.
- First, identify the core questions in the query.
- Second, extract relevant facts and evidence from each source context.
- Third, critically evaluate the evidence, noting quality and potential conflicts.
- Fourth, formulate the key concepts, synthesis insights, and research gaps based on your evaluation.
- Finally, construct the complete JSON output based on your reasoning.

VI. Quality Standards

- Be comprehensive yet focused on answering the specific query
- Demonstrate sophisticated critical thinking and analysis
- Generate novel insights through rigorous synthesis
- Maintain academic credibility with evidence-based reasoning
- Provide practical value for {task_description}
"""
        return prompt
    
    def _parse_enhanced_response(
        self, 
        response_text: str, 
        contexts: List[RetrievedContext],
        thoughts: List[str]
    ) -> Optional[SynthesizedKnowledge]:
        """Parse Gemini response into enhanced SynthesizedKnowledge object."""
        
        try:
            response_json = json.loads(response_text)
            
            # Parse key concepts with enhanced structure
            key_concepts = []
            for concept_data in response_json.get("key_concepts", []):
                try:
                    evidence_quality = None
                    if concept_data.get("evidence_quality"):
                        evidence_quality = EvidenceQuality(concept_data["evidence_quality"].lower())
                except ValueError:
                    evidence_quality = EvidenceQuality.INSUFFICIENT
                
                key_concept = KeyConcept(
                    concept=concept_data.get("concept", "Unknown Concept"),
                    explanation=concept_data.get("explanation", ""),
                    importance=concept_data.get("importance", ""),
                    evidence_quality=evidence_quality,
                    controversies=concept_data.get("controversies"),
                    related_concepts=concept_data.get("related_concepts", []),
                    confidence_score=concept_data.get("confidence_score")
                )
                key_concepts.append(key_concept)
            
            # Parse synthesis insights
            synthesis_insights = []
            for insight_data in response_json.get("synthesis_insights", []):
                insight = SynthesisInsight(
                    insight=insight_data.get("insight", ""),
                    supporting_evidence=insight_data.get("supporting_evidence", []),
                    confidence_level=insight_data.get("confidence_level"),
                    implications=insight_data.get("implications")
                )
                synthesis_insights.append(insight)
            
            # Parse research gaps
            research_gaps = []
            for gap_data in response_json.get("research_gaps", []):
                gap = ResearchGap(
                    gap_description=gap_data.get("gap_description", ""),
                    severity=gap_data.get("severity"),
                    suggested_investigation=gap_data.get("suggested_investigation")
                )
                research_gaps.append(gap)
            
            # Determine analysis depth
            analysis_depth = AnalysisDepth.PHD_LEVEL
            if key_concepts and synthesis_insights:
                analysis_depth = AnalysisDepth.EXPERT
            elif key_concepts or synthesis_insights:
                analysis_depth = AnalysisDepth.PHD_LEVEL
            else:
                analysis_depth = AnalysisDepth.STANDARD
            
            # Create enhanced SynthesizedKnowledge object
            return SynthesizedKnowledge(
                summary=response_json.get("summary", ""),
                key_concepts=key_concepts,
                topics=response_json.get("topics", []),
                synthesis_insights=synthesis_insights,
                research_gaps=research_gaps,
                methodological_observations=response_json.get("methodological_observations"),
                theoretical_implications=response_json.get("theoretical_implications"),
                analysis_depth=analysis_depth,
                overall_confidence=response_json.get("overall_confidence"),
                completeness_score=response_json.get("completeness_score"),
                source_chunk_ids=[ctx.chunk_id for ctx in contexts],
                num_source_contexts=len(contexts),
                analysis_timestamp=datetime.now(),
                analysis_model=self.model_name,
                query_complexity=response_json.get("query_complexity"),
                synthesis_quality_indicators=response_json.get("synthesis_quality_indicators", {}),
                analysis_thoughts=thoughts
            )
            
        except json.JSONDecodeError:
            logger.warning("Gemini response is not valid JSON. Creating fallback synthesis.")
            return SynthesizedKnowledge(
                summary=response_text[:1000] if len(response_text) > 1000 else response_text,
                source_chunk_ids=[ctx.chunk_id for ctx in contexts],
                analysis_depth=AnalysisDepth.FALLBACK_TEXT,
                num_source_contexts=len(contexts),
                analysis_model=self.model_name,
                analysis_thoughts=thoughts
            )
        except Exception as e:
            logger.error(f"Error parsing enhanced response: {str(e)}")
            return SynthesizedKnowledge(
                summary=f"Error parsing synthesis response: {str(e)}",
                source_chunk_ids=[ctx.chunk_id for ctx in contexts],
                analysis_depth=AnalysisDepth.ERROR,
                num_source_contexts=len(contexts),
                analysis_model=self.model_name,
                analysis_thoughts=thoughts
            )
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about the enhanced analyzer configuration."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "api_available": self.client is not None,
            "analysis_capabilities": [
                "phd_level_analysis",
                "synthesis_insights", 
                "research_gap_identification",
                "evidence_quality_assessment",
                "confidence_scoring"
            ]
        }
