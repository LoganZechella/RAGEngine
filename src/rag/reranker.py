"""
Re-Ranker for RAG Engine.
Re-ranks search results for improved relevance using LLM-based evaluation.
Extracted from APEGA with full functionality preserved.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
from openai import OpenAI

from src.models.data_models import RetrievedContext


class ReRanker:
    """
    Re-ranks search results for improved relevance using LLM-based evaluation.
    Uses OpenAI's o4-mini model to score the relevance of retrieved contexts.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "o4-mini-2025-04-16",
        temperature: float = 1,
        verbose: bool = False
    ):
        """
        Initialize the ReRanker.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model_name: Name of the OpenAI model to use
            temperature: Temperature for the model (lower for more consistent scoring)
            verbose: Whether to log detailed output
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it via parameter or OPENAI_API_KEY environment variable.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
    
    def rerank_contexts(
        self, 
        query_text: str, 
        contexts: List[RetrievedContext], 
        top_n: int = 5
    ) -> List[RetrievedContext]:
        """
        Re-rank search results based on relevance to the query.
        
        Args:
            query_text: The original query text
            contexts: List of retrieved contexts to re-rank
            top_n: Number of top results to return after re-ranking
            
        Returns:
            Re-ranked list of RetrievedContext objects
        """
        if not contexts:
            logger.warning("No contexts to re-rank")
            return []
        
        logger.info(f"Re-ranking {len(contexts)} contexts")
        
        # Relevance scoring using LLM
        scored_contexts = self._score_contexts_with_llm(query_text, contexts)
        
        # Sort by LLM-assigned relevance score
        sorted_contexts = sorted(scored_contexts, key=lambda x: x.rerank_score or 0, reverse=True)
        
        # Return top N results
        top_contexts = sorted_contexts[:top_n]
        
        logger.info(f"Re-ranking complete. Selected top {len(top_contexts)} contexts.")
        return top_contexts
    
    def _score_contexts_with_llm(
        self, 
        query_text: str, 
        contexts: List[RetrievedContext]
    ) -> List[RetrievedContext]:
        """
        Score contexts for relevance using an LLM.
        
        Args:
            query_text: The original query text
            contexts: List of retrieved contexts to score
            
        Returns:
            List of RetrievedContext objects with rerank_score set
        """
        scored_contexts = []
        
        for i, context in enumerate(contexts):
            try:
                # Create prompt for the LLM to evaluate relevance
                prompt = self._create_scoring_prompt(query_text, context.text)
                
                # Call OpenAI API with improved handling for o4-mini structured output issues
                try:
                    # Use regular completion instead of structured output to avoid empty response issue
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that evaluates text relevance. Respond with a score between 0.0 and 1.0 followed by a brief justification."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_completion_tokens=10000,  # Reduced to avoid length limit issues
                        timeout=60
                    )
                    
                    # Validate response exists and has content
                    if not response or not response.choices or not response.choices[0].message:
                        logger.warning(f"Invalid response structure for context {i}. Setting score to 0.")
                        context.rerank_score = 0.0
                        scored_contexts.append(context)
                        continue
                        
                except Exception as api_error:
                    logger.error(f"OpenAI API error for context {i} with model '{self.model_name}': {api_error}")
                    # Set a very low but non-zero score to preserve some ranking
                    context.rerank_score = 0.01
                    scored_contexts.append(context)
                    continue
                
                # Extract and parse score from natural language response
                response_text = response.choices[0].message.content
                
                if self.verbose:
                    logger.debug(f"Response for context {i}: {response_text}")
                
                # Parse score from natural language response (more reliable than JSON)
                try:
                    # Check for empty or whitespace-only responses
                    if not response_text or not response_text.strip():
                        logger.warning(f"Empty response from model for context {i}. Setting score to 0.")
                        score = 0.0
                    else:
                        # Extract score using regex pattern matching
                        import re
                        
                        # Look for score patterns like "0.8", "Score: 0.7", "8/10", etc.
                        score_patterns = [
                            r'(?:score[:\s]*)?([0-1]\.?\d*)',  # "score: 0.8" or "0.8"
                            r'(\d+)/10',  # "8/10" format
                            r'(\d+)%'     # "80%" format
                        ]
                        
                        score = 0.0
                        for pattern in score_patterns:
                            match = re.search(pattern, response_text.lower())
                            if match:
                                raw_score = float(match.group(1))
                                # Convert different formats to 0-1 scale
                                if '/10' in pattern:
                                    score = raw_score / 10.0
                                elif '%' in pattern:
                                    score = raw_score / 100.0
                                else:
                                    score = raw_score
                                break
                        
                        # Ensure score is within valid 0-1 range
                        score = max(0.0, min(1.0, score))
                        
                        if self.verbose:
                            logger.debug(f"Extracted score {score} from response: {response_text[:100]}...")
                            
                except Exception as e:
                    logger.warning(f"Error parsing score for context {i}: {e}. Response was: '{response_text[:100]}...'. Setting score to 0.")
                    score = 0.0
                
                # Update context with re-ranking score
                context.rerank_score = score
                scored_contexts.append(context)
                
                if self.verbose:
                    logger.debug(f"Scored context {i}: {score}")
                
            except Exception as e:
                logger.error(f"Error scoring context {i}: {str(e)}")
                # If scoring fails, keep the original context but with a low score
                context.rerank_score = 0.01
                scored_contexts.append(context)
        
        return scored_contexts
    
    def _create_scoring_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM to score the relevance of a context to a query.
        
        Args:
            query: The query text
            context: The context text to evaluate
            
        Returns:
            Prompt for the LLM
        """
        # Truncate context if too long to avoid token limit issues
        max_context_length = 10000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            
        return f"""Rate how relevant this text is to answering the query.

Query: {query}

Text: {context}

Rate from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).
Give your score and brief reason.

Score:"""

    def get_reranker_info(self) -> Dict[str, Any]:
        """
        Get information about the reranker configuration.
        
        Returns:
            Dictionary containing reranker configuration and status
        """
        return {
            "model": self.model_name,
            "max_tokens": 10000, 
            "temperature": self.temperature,
            "api_available": self.client is not None
        }