"""
Deep Analyzer for RAG Engine.
Performs PhD-level comprehensive analysis of retrieved content using Google's Gemini 2.5 Pro.
Utilizes a sophisticated prompt system for rigorous academic-grade knowledge synthesis,
critical analysis, and novel insight generation. Extracted from APEGA with generalizations 
for broader use cases.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
import google.generativeai as genai

from src.models.data_models import RetrievedContext, SynthesizedKnowledge


class DeepAnalyzer:
    """
    Performs PhD-level comprehensive analysis of retrieved content using Google's Gemini 2.5 Pro.
    
    Utilizes a sophisticated prompt system designed for academic-grade analysis, featuring:
    - Critical analysis and evaluation of source material
    - Novel synthesis connecting disparate concepts
    - Identification of research gaps and controversies
    - Methodological assessment and theoretical implications
    - Rigorous reasoning with Chain-of-Thought approaches
    
    Synthesizes knowledge from retrieved contexts to provide deep, scholarly understanding
    suitable for academic research and professional knowledge work.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro-preview-05-06",
        temperature: float = 0.2,
        max_output_tokens: int = 20000,
        verbose: bool = False
    ):
        """
        Initialize the DeepAnalyzer with PhD-level analysis capabilities.
        
        Args:
            api_key: Google API key (defaults to environment variable)
            model_name: Name of the Google Gemini model to use (default: gemini-2.5-pro-preview-05-06)
            temperature: Temperature for the model (default: 0.2 for more focused analysis)
            max_output_tokens: Maximum number of output tokens (default: 16384 for comprehensive analysis)
            verbose: Whether to log detailed output
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set it via parameter or GOOGLE_API_KEY environment variable.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.verbose = verbose
        
        # Initialize Google Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json"
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
    
    def synthesize_knowledge(
        self,
        query_details: Dict[str, Any],
        contexts: List[RetrievedContext]
    ) -> SynthesizedKnowledge:
        """
        Perform PhD-level knowledge synthesis from retrieved contexts.
        
        Employs a comprehensive academic analysis approach that goes beyond simple
        summarization to provide critical evaluation, novel insights, and scholarly
        synthesis of the provided information.
        
        Args:
            query_details: Details about the query (e.g., domain, task, query text)
            contexts: List of retrieved and re-ranked contexts to analyze
            
        Returns:
            SynthesizedKnowledge object with comprehensive academic-grade analysis including:
            - Critical summary with analytical insights
            - Enhanced key concepts with evidence assessment
            - Novel synthesis insights and connections
            - Identified research gaps and controversies
            - Methodological observations
            - Theoretical implications
        """
        if not contexts:
            logger.warning("No contexts provided for knowledge synthesis")
            return SynthesizedKnowledge(
                summary="No context available for synthesis.",
                source_chunk_ids=[]
            )
        
        logger.info(f"Synthesizing knowledge from {len(contexts)} contexts")
        
        # Extract text from contexts and create a consolidated input
        context_texts = [f"Context {i+1}:\n{ctx.text}\n" for i, ctx in enumerate(contexts)]
        consolidated_context = "\n\n".join(context_texts)
        
        # Create prompt for Gemini
        prompt = self._create_synthesis_prompt(query_details, consolidated_context)
        
        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            if self.verbose:
                logger.debug(f"Gemini response: {response.text}")
            
            # Parse the response into a SynthesizedKnowledge object
            synthesized = self._parse_synthesis_response(response.text, contexts)
            
            logger.info(f"Successfully synthesized knowledge with {len(synthesized.key_concepts)} key concepts")
            return synthesized
            
        except Exception as e:
            logger.error(f"Error in knowledge synthesis: {str(e)}")
            return None
    
    def _create_synthesis_prompt(self, query_details: Dict[str, Any], consolidated_context: str) -> str:
        """
        Create a comprehensive PhD-level analysis prompt for Gemini to synthesize knowledge.
        
        Args:
            query_details: Details about the query
            consolidated_context: Combined text from all contexts
            
        Returns:
            Prompt for Gemini
        """
        # Extract relevant details
        domain = query_details.get("domain", "")
        task_description = query_details.get("task", "Analyze and synthesize information")
        query = query_details.get("query", "")
        
        # Build the comprehensive PhD-level prompt with detailed instructions
        prompt = f"""
I. Master Directive & Persona Definition

A. Persona Activation:

Adopt the persona of an Expert Multidisciplinary PhD-Level Researcher and Critical Analyst. You possess deep knowledge of research methodologies, critical thinking, information synthesis, and academic writing conventions across various fields. Your primary objective is to produce scholarly work of the highest caliber, characterized by originality in synthesis, rigorous analysis, and impeccable sourcing. This persona is not merely for stylistic purposes; it is intended to activate the sophisticated knowledge and complex reasoning patterns associated with high-level academic discourse within your training data, ensuring the depth and quality of your output.

B. Overarching Goal Definition:

Your ultimate task is to generate a PhD-level academic analysis and synthesis of the provided information related to the query: {query}
{f"Domain: {domain}" if domain else ""}

This analysis must be exhaustive, demonstrating a profound understanding of the subject, critical engagement with the provided literature, methodologically sound analytical practices, and the ability to synthesize information into novel insights and coherent arguments. The analysis must be suitable for inclusion in a peer-reviewed academic publication.

C. Interpretation of "PhD-Level Quality":

Throughout this task, "PhD-Level Quality" is a guiding framework for all your actions and serves as a multi-dimensional rubric against which your process and output will be constantly evaluated. It implies:
- Critical Analysis: You must move beyond mere summarization. Your analysis involves evaluating, questioning, and interpreting information. This includes identifying underlying assumptions, potential biases, methodological strengths, and crucial weaknesses within the sources you analyze.
- Novel Synthesis: You are expected to connect disparate pieces of information, even from seemingly unrelated subfields if relevant, to generate new understanding. This involves identifying overarching themes, or proposing new perspectives or conceptual frameworks that are not explicitly stated in the source material but emerge from a deep and comprehensive analysis.
- Methodological Rigor & Transparency: Your analysis process must be transparent and replicable in principle. You will clearly articulate the analytical methods employed to arrive at your conclusions.
- Depth of Understanding: You must demonstrate a comprehensive and nuanced grasp of the topic, including its complexities, historical context, current debates, and potential future trajectories.
- Robust Argumentation: All claims must be supported by credible evidence. You will construct well-supported arguments, proactively identify and address potential counter-arguments, and provide logical, evidence-based refutations or qualifications where necessary.

II. Core Analysis Process

A. Deep Information Synthesis & Novel Analysis:

The following text segments have been retrieved as relevant to your analysis:

{consolidated_context}

Your task is to synthesize this information into a coherent, insightful, and critically informed analysis.

- Organize the collected information logically, structuring it around key themes and patterns that emerge from the literature.
- Identify and clearly articulate patterns, trends, significant findings, and emerging consensus that arise from the body of information.
- Crucially, you must also identify and discuss gaps in the research, inconsistencies between sources, contradictions in findings, or unresolved debates and controversies within the current body of knowledge.

B. Advanced Reasoning Requirements:

Go beyond summarization to achieve Novel Synthesis and Critical Analysis:
- Connect disparate ideas: Actively seek to draw connections between findings, theories, or concepts from different sources that may not be explicitly linked in the literature you have reviewed.
- Identify underlying principles or frameworks: Look for deeper patterns. Can you abstract common themes or conceptual structures that help explain the observed phenomena or organize the diverse findings?
- Offer new interpretations or perspectives: Based on your comprehensive and critical understanding of the literature, propose fresh ways of looking at the topic.
- Construct nuanced arguments: Develop clear central insights supported by robust, multi-faceted evidence drawn from your analysis. Proactively identify potential counter-arguments or alternative explanations.

C. Employ Advanced Reasoning Techniques:

- Utilize Chain-of-Thought (CoT) reasoning internally for complex analytical tasks. When analyzing a complex issue or synthesizing multiple viewpoints, explicitly break down your process.
- For identifying research gaps or areas of controversy, employ systematic questioning: What does the current body of literature definitively establish? What are the explicitly stated limitations? What related questions or aspects remain unanswered or underexplored?

III. Required Output Format

Provide your synthesized knowledge in JSON format with the following structure, ensuring each section meets PhD-level standards:

{{
  "summary": "<A comprehensive, critical summary of the core knowledge that goes beyond mere description to provide analytical insights, identify key patterns, and highlight areas of significance or controversy (2-3 paragraphs)>",
  "key_concepts": [
    {{
      "concept": "<Name or title of the concept>",
      "explanation": "<Clear, precise explanation that demonstrates deep understanding>",
      "importance": "<Why this is important, its implications, and its relationship to other concepts>",
      "evidence_quality": "<Assessment of the strength of evidence supporting this concept>",
      "controversies": "<Any debates, limitations, or alternative perspectives related to this concept>"
    }},
    ...
  ],
  "topics": [
    "<Important topic or area that emerges from the analysis, including emerging themes and research directions>",
    ...
  ],
  "synthesis_insights": [
    "<Novel insights or connections that emerge from cross-referencing the provided information>",
    ...
  ],
  "research_gaps": [
    "<Identified gaps, limitations, or areas requiring further investigation>",
    ...
  ],
  "methodological_observations": "<Critical assessment of the methodological approaches evident in the analyzed content>",
  "theoretical_implications": "<Broader theoretical or practical implications that emerge from the synthesis>"
}}

IV. Quality Standards

Ensure your analysis is:
- Comprehensive yet focused, emphasizing content that would be valuable for {task_description}
- Critically rigorous, moving beyond surface-level description to deep analytical insight
- Methodologically transparent in your reasoning process
- Novel in its synthesis, identifying connections and patterns not explicitly stated in individual sources
- Academically credible, with all claims supported by evidence from the provided contexts

Your output should represent the caliber of analysis expected in top-tier academic research, demonstrating sophisticated critical thinking and the ability to generate new insights through rigorous synthesis.
"""
        return prompt
    
    def _parse_synthesis_response(
        self, 
        response_text: str, 
        contexts: List[RetrievedContext]
    ) -> SynthesizedKnowledge:
        """
        Parse the Gemini response into a SynthesizedKnowledge object.
        
        Args:
            response_text: Response from Gemini
            contexts: Original contexts used for synthesis
            
        Returns:
            SynthesizedKnowledge object
        """
        try:
            # Try to parse as JSON
            import json
            response_json = json.loads(response_text)
            
            # Extract fields from the response
            summary = response_json.get("summary", "")
            key_concepts = response_json.get("key_concepts", [])
            topics = response_json.get("topics", [])
            
            # Extract additional fields from the enhanced PhD-level analysis
            synthesis_insights = response_json.get("synthesis_insights", [])
            research_gaps = response_json.get("research_gaps", [])
            methodological_observations = response_json.get("methodological_observations", "")
            theoretical_implications = response_json.get("theoretical_implications", "")
            
            # Create enhanced metadata with additional analysis fields
            enhanced_metadata = {
                "synthesis_insights": synthesis_insights,
                "research_gaps": research_gaps,
                "methodological_observations": methodological_observations,
                "theoretical_implications": theoretical_implications,
                "analysis_depth": "phd_level",
                "num_source_contexts": len(contexts)
            }
            
            # Create SynthesizedKnowledge object with enhanced metadata
            return SynthesizedKnowledge(
                summary=summary,
                key_concepts=key_concepts,
                topics=topics,
                source_chunk_ids=[ctx.chunk_id for ctx in contexts],
                metadata=enhanced_metadata
            )
            
        except json.JSONDecodeError:
            # If not valid JSON, use the text as-is for the summary
            logger.warning("Gemini response is not valid JSON. Using as plain text.")
            return SynthesizedKnowledge(
                summary=response_text,
                source_chunk_ids=[ctx.chunk_id for ctx in contexts],
                metadata={"analysis_depth": "fallback_text", "num_source_contexts": len(contexts)}
            )
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            return SynthesizedKnowledge(
                summary=f"Error parsing synthesis response: {str(e)}",
                source_chunk_ids=[ctx.chunk_id for ctx in contexts],
                metadata={"analysis_depth": "error", "num_source_contexts": len(contexts)}
            )
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """
        Get information about the deep analyzer configuration.
        
        Returns:
            Dictionary containing analyzer configuration and status
        """
        return {
            "model": self.model_name,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "api_available": self.model is not None
        }