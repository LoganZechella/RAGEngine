"""
Re-Ranker for RAG Engine.
Re-ranks search results for improved relevance using LLM-based evaluation.
Uses OpenAI's GPT-4.1 model for superior instruction following and context understanding.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import os
import time
import re
from openai import OpenAI

from backend.src.models.data_models import RetrievedContext


class ReRanker:
    """
    Re-ranks search results for improved relevance using LLM-based evaluation.
    Uses OpenAI's GPT-4.1 model for superior instruction following and large context windows.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4.1",
        temperature: float = 0.2,  # Lower temperature for consistent scoring
        max_tokens: int = 500,  # Increased for GPT-4.1's capabilities
        top_p: float = 0.9,  # Slightly focused sampling
        verbose: bool = False
    ):
        """
        Initialize the ReRanker with GPT-4.1.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model_name: Name of the OpenAI model to use (gpt-4.1)
            temperature: Temperature for the model (0.0-2.0, lower for more consistent scoring)
            max_tokens: Maximum tokens for response (GPT-4.1 supports much larger context)
            top_p: Top-p sampling parameter (0.0-1.0)
            verbose: Whether to log detailed output
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it via parameter or OPENAI_API_KEY environment variable.")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.verbose = verbose
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=60.0  # Increased timeout for GPT-4.1
        )
        
        # Rate limiting
        self.last_api_call = 0
        self.min_interval = 0.1  # Minimum seconds between API calls
        
        logger.info(f"Initialized ReRanker with model={model_name}, temperature={temperature}, max_tokens={max_tokens}")
    
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
        
        if not query_text or not query_text.strip():
            logger.warning("Empty query text provided for reranking")
            return contexts[:top_n]  # Return original order if no query
        
        logger.info(f"Re-ranking {len(contexts)} contexts for query: '{query_text[:50]}...'")
        
        # Relevance scoring using GPT-4.1
        scored_contexts = self._score_contexts_with_llm(query_text, contexts)
        
        # Sort by LLM-assigned relevance score (descending)
        sorted_contexts = sorted(scored_contexts, key=lambda x: x.rerank_score or 0, reverse=True)
        
        # Return top N results
        top_contexts = sorted_contexts[:top_n]
        
        if self.verbose:
            logger.info("Reranking results:")
            for i, ctx in enumerate(top_contexts):
                logger.info(f"  {i+1}. Score: {ctx.rerank_score:.3f} - {ctx.text[:100]}...")
        
        logger.info(f"Re-ranking complete. Selected top {len(top_contexts)} contexts.")
        return top_contexts
    
    def _rate_limit(self):
        """Simple rate limiting to avoid hitting API limits."""
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_api_call = time.time()
    
    def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Call OpenAI GPT-4.1 API with improved error handling.
        
        Args:
            messages: List of message dictionaries for the API
            
        Returns:
            Response content from the API
        """
        self._rate_limit()
        
        try:
            # Use the Chat Completions API with GPT-4.1 parameters
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
            
            # Validate response structure
            if not response or not response.choices or not response.choices[0].message:
                raise ValueError("Invalid response structure from OpenAI API")
            
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response content from OpenAI API")
            
            return content.strip()
            
        except Exception as e:
            # Log the actual error details for debugging
            error_msg = f"OpenAI API call failed: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _score_contexts_with_llm(
        self, 
        query_text: str, 
        contexts: List[RetrievedContext]
    ) -> List[RetrievedContext]:
        """
        Score contexts for relevance using GPT-4.1.
        
        Args:
            query_text: The original query text
            contexts: List of retrieved contexts to score
            
        Returns:
            List of RetrievedContext objects with rerank_score set
        """
        scored_contexts = []
        
        for i, context in enumerate(contexts):
            try:
                # Create prompt for GPT-4.1 to evaluate relevance
                prompt = self._create_scoring_prompt(query_text, context.text)
                
                # Prepare messages with improved system prompt for GPT-4.1
                messages = [
                    {
                        "role": "system", 
                        "content": """You are an expert relevance evaluator. Your task is to score how relevant a given text passage is to answering a specific query.

Instructions:
1. Read the query carefully to understand what information is being sought
2. Analyze the text passage for relevant information
3. Assign a relevance score from 0.0 to 1.0 where:
   - 0.0 = Completely irrelevant (no useful information)
   - 0.1-0.3 = Minimally relevant (tangentially related)
   - 0.4-0.6 = Moderately relevant (some useful information)
   - 0.7-0.9 = Highly relevant (directly addresses query)
   - 1.0 = Perfectly relevant (fully answers the query)

Respond with ONLY the numeric score followed by a brief justification."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
                
                # Call OpenAI API with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response_text = self._call_openai_api(messages)
                        
                        if self.verbose:
                            logger.debug(f"GPT-4.1 response for context {i}: {response_text}")
                        
                        # Extract and parse score from response
                        score = self._extract_score_from_response(response_text, i)
                        
                        # Update context with re-ranking score
                        context.rerank_score = score
                        scored_contexts.append(context)
                        
                        if self.verbose:
                            logger.debug(f"Scored context {i}: {score:.3f}")
                        
                        break  # Success, exit retry loop
                        
                    except Exception as api_error:
                        logger.error(f"GPT-4.1 API error for context {i}, attempt {attempt + 1}/{max_retries}: {api_error}")
                        
                        if attempt == max_retries - 1:  # Last attempt failed
                            # Set a very low but non-zero score to preserve some ranking
                            context.rerank_score = 0.01
                            scored_contexts.append(context)
                        else:
                            # Wait before retry with exponential backoff
                            time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Error scoring context {i}: {str(e)}")
                # If scoring fails, keep the original context but with a low score
                context.rerank_score = 0.01
                scored_contexts.append(context)
        
        return scored_contexts
    
    def _extract_score_from_response(self, response_text: str, context_index: int) -> float:
        """
        Extract relevance score from GPT-4.1 response using comprehensive regex patterns.
        
        Args:
            response_text: Response from GPT-4.1
            context_index: Index of the context being scored (for logging)
            
        Returns:
            Extracted score between 0.0 and 1.0
        """
        if not response_text or not response_text.strip():
            logger.warning(f"Empty response for context {context_index}")
            return 0.0
        
        try:
            # Enhanced regex patterns optimized for GPT-4.1 responses
            score_patterns = [
                # Direct decimal scores at start: "0.8", "0.75"
                r'^([01]\.?\d*)',
                # Score with labels: "Score: 0.8", "Relevance: 0.7"
                r'(?:score|relevance|rating)\s*[:\-]\s*([01]\.?\d*)',
                # Fraction format: "8/10", "7.5/10"
                r'(\d+(?:\.\d+)?)\s*/\s*10',
                # Percentage format: "80%", "75%"
                r'(\d+(?:\.\d+)?)\s*%',
                # Decimal anywhere in response: find any valid 0.x number
                r'([01]\.\d+)',
                # Integer scores: "8" or "7" (convert to 0.x)
                r'^([0-9])\s'
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, response_text.lower())
                if matches:
                    try:
                        raw_score = float(matches[0])
                        
                        # Convert different formats to 0-1 scale
                        if '/10' in pattern:
                            score = raw_score / 10.0
                        elif '%' in pattern:
                            score = raw_score / 100.0
                        elif pattern.endswith(r'^([0-9])\s'):  # Single digit
                            score = raw_score / 10.0
                        else:
                            score = raw_score
                        
                        # Ensure score is within valid 0-1 range
                        score = max(0.0, min(1.0, score))
                        
                        if self.verbose:
                            logger.debug(f"Extracted score {score:.3f} using pattern from: {response_text[:100]}...")
                        
                        return score
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing score with pattern '{pattern}': {e}")
                        continue
            
            # Enhanced fallback: look for any number that could be a score
            all_numbers = re.findall(r'\d+\.?\d*', response_text)
            for num_str in all_numbers:
                try:
                    num = float(num_str)
                    if 0 <= num <= 1:
                        logger.debug(f"Using fallback score {num} from response: {response_text[:100]}...")
                        return num
                    elif 1 < num <= 10:
                        score = num / 10.0
                        logger.debug(f"Using fallback score {score} (converted from {num}/10) from response: {response_text[:100]}...")
                        return score
                    elif 10 < num <= 100:
                        score = num / 100.0
                        logger.debug(f"Using fallback score {score} (converted from {num}%) from response: {response_text[:100]}...")
                        return score
                except ValueError:
                    continue
            
            logger.warning(f"Could not extract score from GPT-4.1 response for context {context_index}: '{response_text[:200]}...'")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error extracting score for context {context_index}: {e}")
            return 0.0
    
    def _create_scoring_prompt(self, query: str, context: str) -> str:
        """
        Create an optimized prompt for GPT-4.1 to score relevance.
        
        Args:
            query: The query text
            context: The context text to evaluate
            
        Returns:
            Prompt optimized for GPT-4.1
        """
        # GPT-4.1 can handle much larger contexts (up to 1M tokens)
        max_context_length = 8000  # Increased for GPT-4.1's capabilities
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Truncate query if extremely long
        max_query_length = 1000  # Increased for complex queries
        if len(query) > max_query_length:
            query = query[:max_query_length] + "..."
            
        return f"""QUERY: {query}

TEXT TO EVALUATE: {context}

Task: Rate how relevant this text is to answering the query above.

Scoring Scale:
- 0.0: Completely irrelevant
- 0.1-0.3: Minimally relevant 
- 0.4-0.6: Moderately relevant
- 0.7-0.9: Highly relevant
- 1.0: Perfectly relevant

Provide your score as a decimal number between 0.0 and 1.0, followed by a brief explanation.

Example: "0.8 - This text directly addresses the main question about X and provides specific details."

Your evaluation:"""

    def get_reranker_info(self) -> Dict[str, Any]:
        """
        Get information about the reranker configuration.
        
        Returns:
            Dictionary containing reranker configuration and status
        """
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "api_available": self.client is not None,
            "rate_limit_interval": self.min_interval,
            "context_window": "1M tokens"  # GPT-4.1 capability
        }