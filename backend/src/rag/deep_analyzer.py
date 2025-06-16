"""
Enhanced Deep Analyzer for RAG Engine.
Performs PhD-level comprehensive analysis with improved output structure and readability.
Optimized for the enhanced SynthesizedKnowledge data model.
Supports both Gemini and OpenAI o3 models with configuration-based switching.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
import json
from google import genai
from google.genai import types
from datetime import datetime

# Conditional OpenAI import - lazy loading to avoid dependency issues
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Only Gemini provider will be supported.")

from backend.src.models.data_models import (
    RetrievedContext, SynthesizedKnowledge, KeyConcept, SynthesisInsight, 
    ResearchGap, AnalysisDepth, EvidenceQuality
)


class DeepAnalyzer:
    """
    Enhanced Deep Analyzer for PhD-level comprehensive analysis with dual provider support.
    Supports both Gemini 2.5-pro and OpenAI o3 models with configuration-based switching.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro-preview-06-05",
        model_provider: str = "gemini",  # FIXED: Back to "gemini" for backwards compatibility
        openai_api_key: Optional[str] = None,  # NEW
        openai_model: str = "o3-2025-04-16",  # NEW
        reasoning_effort: str = "medium",  # NEW: "low", "medium", "high"
        temperature: float = 0.2,
        max_output_tokens: int = 20000,
        verbose: bool = False
    ):
        """Initialize the Enhanced DeepAnalyzer with dual provider support."""
        
        # Environment variable override for provider selection
        self.model_provider = os.getenv("MODEL_PROVIDER", model_provider).lower()
        
        # Common configuration
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.verbose = verbose
        
        # Initialize based on provider
        if self.model_provider == "gemini":
            self._init_gemini(api_key, model_name)
        elif self.model_provider == "openai":
            self._init_openai(openai_api_key, openai_model, reasoning_effort)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}. Supported: 'gemini', 'openai'")
    
    def _init_gemini(self, api_key: Optional[str], model_name: str):
        """Initialize Gemini provider configuration."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set it via parameter or GOOGLE_API_KEY environment variable.")
        
        self.model_name = model_name
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
    
    def _init_openai(self, openai_api_key: Optional[str], openai_model: str, reasoning_effort: str):
        """Initialize OpenAI provider configuration."""
        if not OPENAI_AVAILABLE:
            raise ValueError("OpenAI library not available. Install with: pip install openai")
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set it via parameter or OPENAI_API_KEY environment variable.")
        
        self.openai_model = os.getenv("OPENAI_MODEL", openai_model)
        self.reasoning_effort = os.getenv("REASONING_EFFORT", reasoning_effort)
        
        # Validate reasoning effort
        if self.reasoning_effort not in ["low", "medium", "high"]:
            logger.warning(f"Invalid reasoning effort '{self.reasoning_effort}', defaulting to 'medium'")
            self.reasoning_effort = "medium"
        
        # OpenAI o3 requires longer timeouts due to reasoning process
        # Set timeout to 3 minutes (180 seconds) to accommodate complex reasoning
        self.openai_client = OpenAI(
            api_key=self.openai_api_key,
            timeout=180.0  # 3 minutes timeout for o3 reasoning
        )
        
        # System instruction for OpenAI
        self.openai_system_instruction = """
Adopt the persona of a PhD-level researcher and critical analyst. 
Your task is to perform a comprehensive analysis of the provided source material.
Use your advanced reasoning capabilities to provide deep, structured insights.
"""

    def synthesize_knowledge(
        self,
        query_details: Dict[str, Any],
        contexts: List[RetrievedContext]
    ) -> Optional[SynthesizedKnowledge]:
        """
        Perform enhanced PhD-level knowledge synthesis with structured output.
        Routes to appropriate provider based on configuration.
        """
        if not contexts:
            logger.warning("No contexts provided for knowledge synthesis")
            return SynthesizedKnowledge(
                summary="No context available for synthesis.",
                source_chunk_ids=[],
                analysis_depth=AnalysisDepth.ERROR,
                num_source_contexts=0
            )
        
        logger.info(f"Synthesizing knowledge from {len(contexts)} contexts using {self.model_provider} provider")
        
        # Route to appropriate provider
        if self.model_provider == "gemini":
            return self._synthesize_with_gemini(query_details, contexts)
        elif self.model_provider == "openai":
            return self._synthesize_with_openai(query_details, contexts)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def _synthesize_with_gemini(
        self,
        query_details: Dict[str, Any],
        contexts: List[RetrievedContext]
    ) -> Optional[SynthesizedKnowledge]:
        """
        Gemini-based knowledge synthesis (existing implementation).
        """
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
                synthesized.analysis_model = self.model_name
                logger.info(f"Successfully synthesized knowledge with {len(synthesized.key_concepts)} concepts, {len(synthesized.synthesis_insights)} insights")
            
            return synthesized
            
        except Exception as e:
            logger.error(f"Error in Gemini knowledge synthesis: {str(e)}")
            return SynthesizedKnowledge(
                summary=f"Error during Gemini synthesis: {str(e)}",
                source_chunk_ids=[ctx.chunk_id for ctx in contexts],
                analysis_depth=AnalysisDepth.ERROR,
                num_source_contexts=len(contexts),
                analysis_model=self.model_name
            )
    
    def _synthesize_with_openai(
        self,
        query_details: Dict[str, Any],
        contexts: List[RetrievedContext]
    ) -> Optional[SynthesizedKnowledge]:
        """
        OpenAI o3 implementation with equivalent functionality to Gemini.
        """
        context_texts = [f"Context {i+1}:\n{ctx.text}\n" for i, ctx in enumerate(contexts)]
        consolidated_context = "\n\n".join(context_texts)
        
        prompt = self._create_enhanced_synthesis_prompt(query_details, consolidated_context)
        
        try:
            logger.info(f"Starting OpenAI o3 synthesis with reasoning_effort='{self.reasoning_effort}' (this may take 30-120 seconds)")
            
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "developer", 
                        "content": self.openai_system_instruction
                    },
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=self.max_output_tokens,
                reasoning_effort=self.reasoning_effort,
                temperature=self.temperature,
            )
            
            logger.info("OpenAI o3 synthesis completed successfully")
            
            # Extract response content and reasoning
            final_response_text = response.choices[0].message.content
            reasoning_tokens = 0
            if response.usage and response.usage.completion_tokens_details:
                reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0
            
            # Convert reasoning tokens to thoughts format for compatibility
            thoughts = [f"Reasoning tokens used: {reasoning_tokens}"] if reasoning_tokens > 0 else []
            
            if self.verbose:
                logger.debug(f"OpenAI reasoning tokens: {reasoning_tokens}")
                logger.debug(f"OpenAI response: {final_response_text}")
            
            # Use same parsing logic as Gemini
            synthesized = self._parse_enhanced_response(final_response_text, contexts, thoughts)
            
            if synthesized:
                # Update model name for tracking
                synthesized.analysis_model = self.openai_model
                logger.info(f"Successfully synthesized knowledge with OpenAI o3: {len(synthesized.key_concepts)} concepts, {len(synthesized.synthesis_insights)} insights")
            
            return synthesized
            
        except Exception as e:
            logger.error(f"Error in OpenAI knowledge synthesis: {str(e)}")
            return SynthesizedKnowledge(
                summary=f"Error during OpenAI synthesis: {str(e)}",
                source_chunk_ids=[ctx.chunk_id for ctx in contexts],
                analysis_depth=AnalysisDepth.ERROR,
                num_source_contexts=len(contexts),
                analysis_model=self.openai_model
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
        """Parse response into enhanced SynthesizedKnowledge object."""
        
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
                analysis_model=getattr(self, 'model_name', None) or getattr(self, 'openai_model', 'unknown'),
                query_complexity=response_json.get("query_complexity"),
                synthesis_quality_indicators=response_json.get("synthesis_quality_indicators", {}),
                analysis_thoughts=thoughts
            )
            
        except json.JSONDecodeError:
            logger.warning("Response is not valid JSON. Creating fallback synthesis.")
            model_name = getattr(self, 'model_name', None) or getattr(self, 'openai_model', 'unknown')
            return SynthesizedKnowledge(
                summary=response_text[:1000] if len(response_text) > 1000 else response_text,
                source_chunk_ids=[ctx.chunk_id for ctx in contexts],
                analysis_depth=AnalysisDepth.FALLBACK_TEXT,
                num_source_contexts=len(contexts),
                analysis_model=model_name,
                analysis_thoughts=thoughts
            )
        except Exception as e:
            logger.error(f"Error parsing enhanced response: {str(e)}")
            model_name = getattr(self, 'model_name', None) or getattr(self, 'openai_model', 'unknown')
            return SynthesizedKnowledge(
                summary=f"Error parsing synthesis response: {str(e)}",
                source_chunk_ids=[ctx.chunk_id for ctx in contexts],
                analysis_depth=AnalysisDepth.ERROR,
                num_source_contexts=len(contexts),
                analysis_model=model_name,
                analysis_thoughts=thoughts
            )
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about the enhanced analyzer configuration with dual provider support."""
        base_info = {
            "provider": self.model_provider,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "analysis_capabilities": [
                "phd_level_analysis",
                "synthesis_insights", 
                "research_gap_identification",
                "evidence_quality_assessment",
                "confidence_scoring"
            ]
        }
        
        if self.model_provider == "gemini":
            base_info.update({
                "model": self.model_name,
                "api_available": self.client is not None,
                "thinking_enabled": True,
                "provider_specific": {
                    "google_api_key_set": bool(self.api_key),
                    "thinking_config": "enabled"
                }
            })
        elif self.model_provider == "openai":
            base_info.update({
                "model": self.openai_model,
                "api_available": self.openai_client is not None,
                "reasoning_effort": self.reasoning_effort,
                "reasoning_enabled": True,
                "provider_specific": {
                    "openai_api_key_set": bool(self.openai_api_key),
                    "reasoning_effort_level": self.reasoning_effort,
                    "openai_library_available": OPENAI_AVAILABLE
                }
            })
        
        return base_info
