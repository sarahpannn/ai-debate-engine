"""
AI Service for Debate Simulation with Financial Document Search Tools
Uses claudette Chat for all Claude interactions.
"""

import asyncio
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from claudette import Chat
from loguru import logger

from llm_personas import LLMPersona
from debate_formats import SpeechType, Speech, DebateFormat
from query_database import create_searcher_from_env, SearchFilters


@dataclass
class DebateContext:
    topic: str
    current_speeches: list[Speech]
    persona: LLMPersona
    side: str  # "government" or "opposition"


def search_financial_documents(
    query: str,
    ticker: Optional[str] = None,
    document_type: Optional[str] = None,
    limit: int = 5,
    min_similarity: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Tool for searching financial documents to support debate arguments.
    
    Args:
        query: Search query for relevant financial information
        ticker: Optional stock symbol to filter by (e.g., "AAPL", "GTLB")
        document_type: Optional document type ("10-K", "10-Q", "earnings_call", "conference")
        limit: Maximum number of results (default: 5)
        min_similarity: Minimum similarity threshold (default: 0.0)
        
    Returns:
        List of search results with content and metadata
    """
    try:
        searcher = create_searcher_from_env()
        
        # Build search filters
        filters = SearchFilters(
            tickers=[ticker.upper()] if ticker else None,
            limit=limit,
            min_similarity=min_similarity
        )
        
        # Handle document type filtering
        if document_type:
            results = searcher.search_by_document_type(query, document_type, limit, min_similarity)
        elif ticker:
            results = searcher.search_by_ticker(query, ticker, limit, min_similarity)
        else:
            results = searcher.search(query, filters)
        
        # Convert to serializable format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                "similarity_score": round(result.similarity_score, 3),
                "source": result.table_name,
                "ticker": result.ticker,
                "title": result.title,
                "filing_type": result.filing_type,
                "chunk_index": result.chunk_index
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error in financial document search: {e}")
        return [{"error": f"Search failed: {str(e)}"}]


class ClaudeDebateService:
    """Claude AI service using claudette Chat with financial document search capabilities"""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import httpx
        from anthropic import Anthropic
        
        # Create a clean HTTP client without proxy configuration
        http_client = httpx.Client()
        
        # Create Anthropic client with clean HTTP client
        anthropic_client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            http_client=http_client
        )
        
        # Create claudette Chat with explicit client and tools
        self.chat = Chat(model, tools=[search_financial_documents])
        # Create a separate Chat instance for judging (without tools to avoid conflicts)
        self.judge_chat = Chat(model)  # No tools for judge
        
        # Store the client for direct access if needed
        self.anthropic_client = anthropic_client
        
        # Enable prompt caching for system prompts to save costs
        # We'll manually set cache_control on system messages
        self.cache_enabled = True
        self.model = model
        self.debate_format = DebateFormat()
        
    def create_fresh_chat(self):
        """Create a new Chat instance for parallel debate isolation"""
        return Chat(self.model, tools=[search_financial_documents])
    
    async def generate_speech(
        self, 
        context: DebateContext,
        speech_type: SpeechType,
        max_retries: int = 3
    ) -> str:
        """Generate a debate speech using claudette Chat with tool access"""
        
        # Create a fresh chat instance for this speech to avoid conflicts in parallel execution
        chat = self.create_fresh_chat()
        
        # Build context from previous speeches
        speech_history = self._format_speech_history(context.current_speeches)
        
        # Create enhanced system prompt with RAG capabilities
        enhanced_system_prompt = self._build_enhanced_system_prompt(
            context.persona, speech_type, context.side
        )
        
        # Create debate-specific user prompt
        user_prompt = self._build_speech_prompt(
            topic=context.topic,
            speech_type=speech_type,
            side=context.side,
            speech_history=speech_history
        )
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating {speech_type.value} speech with tools (attempt {attempt + 1})")
                
                # Set system prompt for this fresh chat instance
                chat.sp = enhanced_system_prompt
                
                # Use toolloop to handle tool calls and get final text response
                toolloop_results = list(chat.toolloop(user_prompt))
                
                # Extract the final text content from toolloop results
                if toolloop_results and len(toolloop_results) > 0:
                    # toolloop_results is a list of tuples from the final message
                    # Find the content tuple which contains the actual text
                    content_data = None
                    for key, value in toolloop_results:
                        if key == 'content':
                            content_data = value
                            break
                    
                    if content_data and isinstance(content_data, list):
                        # Extract text from TextBlocks
                        texts = [block.text for block in content_data if hasattr(block, 'text')]
                        speech_content = " ".join(texts).strip()
                    else:
                        speech_content = "Error: No valid content found in toolloop results"
                else:
                    speech_content = "Error: No toolloop results generated"

                print(f"Generated speech content: {speech_content[:200]}...")  # Debug print first 200 chars
                
                # Validate speech meets format requirements
                is_valid, message = self.debate_format.validate_speech(speech_content)
                if is_valid:
                    logger.success(f"Generated valid {speech_type.value} speech with tool support")
                    return speech_content
                else:
                    logger.warning(f"Invalid speech generated: {message}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        # Truncate if too long, error if too short
                        words = speech_content.split()
                        if len(words) > self.debate_format.max_words:
                            truncated = " ".join(words[:self.debate_format.max_words])
                            logger.info(f"Truncated speech to {self.debate_format.max_words} words")
                            return truncated
                        else:
                            raise ValueError(f"Speech generation failed: {message}")
                            
            except Exception as e:
                logger.error(f"Error generating speech (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError(f"Failed to generate speech after {max_retries} attempts")
    
    def create_fresh_judge_chat(self):
        """Create a new judge Chat instance without tools for parallel isolation"""
        return Chat(self.model)  # No tools for judge
    
    async def judge_debate(
        self, 
        topic: str, 
        speeches: list[Speech],
        max_retries: int = 3
    ) -> tuple[str, str]:
        """Use claudette Chat as judge to determine debate winner"""
        
        judge_prompt = self._build_judge_prompt(topic, speeches)
        
        judge_system = """You are an expert debate judge evaluating a Parliamentary debate.
        
        IMPORTANT: You are judging based ONLY on the speech content provided. Do not use any tools or search for additional information.
        
        Evaluation Criteria:
        1. Argument Quality: Logic, evidence, reasoning, use of factual data
        2. Refutation: How well each side addressed opponent's points
        3. Structure: Organization and flow of arguments
        4. Persuasiveness: Overall convincing power
        5. Evidence Usage: Effective incorporation of financial data and documents
        
        You must declare either "Government" or "Opposition" as the winner.
        Provide clear reasoning for your decision based solely on the speeches presented.
        
        Format your response as:
        WINNER: [Government/Opposition]
        REASONING: [Your detailed analysis]"""
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Judging debate (attempt {attempt + 1})")
                
                # Create fresh judge chat for parallel isolation
                judge_chat = self.create_fresh_judge_chat()
                judge_chat.sp = judge_system
                
                # Use direct chat call for judge (no tools needed)
                response = await asyncio.to_thread(
                    judge_chat,
                    judge_prompt
                )
                
                judgment = str(response).strip()
                winner, reasoning = self._parse_judgment(judgment)
                
                logger.success(f"Debate judged: {winner} wins")
                return winner, reasoning
                
            except Exception as e:
                logger.error(f"Error judging debate (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise RuntimeError(f"Failed to judge debate after {max_retries} attempts")
    
    def _build_enhanced_system_prompt(
        self, 
        persona: LLMPersona, 
        speech_type: SpeechType,
        side: str
    ) -> str:
        """Build enhanced system prompt with RAG database information and caching"""
        
        base_prompt = persona.system_prompt
        
        # Add cache control to the RAG enhancement for cost savings
        rag_enhancement = """
        
        FINANCIAL DOCUMENT DATABASE ACCESS:
        
        You have access to a comprehensive database of financial documents including:
        - SEC 10-K and 10-Q filings
        - Earnings call transcripts
        - Conference presentations
        
        Use the search_financial_documents tool to:
        - Find specific financial data to support your arguments
        - Locate company performance metrics and trends
        - Access market analysis and competitive information
        - Retrieve regulatory filings and compliance data
        
        Tool Usage Guidelines:
        - Search for relevant financial information when making claims about companies or markets
        - Use specific company tickers when available (e.g., "AAPL", "GTLB", "MSFT")
        - Cite sources with company name and document type when referencing data
        - Prioritize recent filings and earnings data for current arguments
        
        Example searches:
        - "revenue growth Q3 2024" with ticker="AAPL"
        - "cybersecurity risks" with document_type="10-K"
        - "AI investment strategy" for general market trends
        
        IMPORTANT: Always fact-check your claims with the database before making specific financial assertions.
        """
        
        # Add cache control to the enhanced prompt
        full_prompt = base_prompt + rag_enhancement
        
        return full_prompt
    
    def _format_speech_history(self, speeches: list[Speech]) -> str:
        """Format previous speeches for context - clean text only to avoid tool conflicts"""
        if not speeches:
            return "This is the opening speech of the debate."
        
        history = "Previous speeches in this debate:\n\n"
        for i, speech in enumerate(speeches, 1):
            side = "Government" if self.debate_format.is_government_speaker(speech.speaker) else "Opposition"
            
            # Aggressively clean the content to remove any tool artifacts
            clean_content = speech.content
            
            # Remove any lines containing tool-related content
            lines = clean_content.split('\n')
            clean_lines = []
            
            for line in lines:
                line_lower = line.lower().strip()
                # Skip lines that contain tool artifacts
                if any(keyword in line_lower for keyword in [
                    'search_financial_documents', 'tool_use', 'toolu_', 
                    '{"query":', '{"ticker":', '"type": "tool_use"',
                    'textblock', 'citations=none'
                ]):
                    continue
                # Skip lines that look like JSON objects
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    continue
                # Skip lines that are mostly brackets or look like artifacts
                if len(line.strip()) < 10 and any(char in line for char in ['[', ']', '{', '}']):
                    continue
                    
                clean_lines.append(line)
            
            clean_content = '\n'.join(clean_lines).strip()
            
            # Further cleanup: remove any remaining tool artifacts at start/end
            while clean_content.startswith(('(', '[', '{')):
                clean_content = clean_content[1:].strip()
            while clean_content.endswith((')', ']', '}')):
                clean_content = clean_content[:-1].strip()
            
            # If content is too short after cleaning, use a placeholder
            if len(clean_content.split()) < 10:
                clean_content = f"[{speech.speaker.value} presented arguments with financial data and analysis]"
            
            history += f"Speech {i} - {speech.speaker.value} ({side}):\n"
            history += f"{clean_content}\n\n"
        
        return history
    
    def _build_speech_prompt(
        self,
        topic: str,
        speech_type: SpeechType,
        side: str,
        speech_history: str
    ) -> str:
        """Build prompt for speech generation"""
        
        role_descriptions = {
            SpeechType.PM: "You are the Prime Minister opening for the Government. Set up your case clearly with strong evidence.",
            SpeechType.LO: "You are the Leader of Opposition. Respond to Government's case and present your data-backed alternative.",
            SpeechType.MG: "You are a Member of Government. Extend and strengthen your side's arguments with financial evidence.",
            SpeechType.MO: "You are a Member of Opposition. Attack Government's case and build Opposition's position with concrete data.",
            SpeechType.OW: "You are Opposition Whip. Summarize why Opposition should win, highlighting key evidence.",
            SpeechType.GW: "You are Government Whip. Summarize why Government should win, reinforcing your strongest data points."
        }
        
        prompt = f"""You are participating in a Parliamentary debate on the topic:
        
        "{topic}"
        
        {role_descriptions[speech_type]}
        
        Your side: {side.title()}
        
        {speech_history}
        
        Requirements:
        - Maximum {self.debate_format.max_words} words
        - Address the topic directly with evidence-based arguments
        - Use the search_financial_documents tool to find supporting data
        - If responding to previous speeches, engage with their arguments
        - Use Parliamentary debate conventions
        - Cite your sources when referencing financial data
        
        Deliver your speech now:"""
        
        return prompt
    
    def _build_judge_prompt(self, topic: str, speeches: list[Speech]) -> str:
        """Build prompt for debate judging"""
        
        prompt = f"""Please judge this Parliamentary debate on the topic: "{topic}"
        
        Here are all six speeches in order:
        
        """
        
        for i, speech in enumerate(speeches, 1):
            side = "Government" if self.debate_format.is_government_speaker(speech.speaker) else "Opposition"
            
            # Clean speech content before showing to judge (same logic as _format_speech_history)
            clean_content = speech.content
            lines = clean_content.split('\n')
            clean_lines = []
            
            for line in lines:
                line_lower = line.lower().strip()
                # Skip lines that contain tool artifacts
                if any(keyword in line_lower for keyword in [
                    'search_financial_documents', 'tool_use', 'toolu_', 
                    '{"query":', '{"ticker":', '"type": "tool_use"',
                    'textblock', 'citations=none'
                ]):
                    continue
                # Skip lines that look like JSON objects
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    continue
                # Skip lines that are mostly brackets or look like artifacts
                if len(line.strip()) < 10 and any(char in line for char in ['[', ']', '{', '}']):
                    continue
                    
                clean_lines.append(line)
            
            clean_content = '\n'.join(clean_lines).strip()
            
            # Further cleanup: remove any remaining tool artifacts at start/end
            while clean_content.startswith(('(', '[', '{')):
                clean_content = clean_content[1:].strip()
            while clean_content.endswith((')', ']', '}')):
                clean_content = clean_content[:-1].strip()
            
            # If content is too short after cleaning, use a placeholder
            if len(clean_content.split()) < 10:
                clean_content = f"[{speech.speaker.value} presented arguments with financial data and analysis]"
            
            prompt += f"Speech {i} - {speech.speaker.value} ({side}):\n"
            prompt += f"{clean_content}\n\n"
        
        prompt += """
        Evaluate the debate based on:
        1. Strength of arguments and evidence (including financial data)
        2. Effective refutation of opponent's points  
        3. Logical structure and flow
        4. Overall persuasiveness
        5. Quality and relevance of cited sources
        
        Who won this debate and why?"""
        
        return prompt
    
    def _parse_judgment(self, judgment: str) -> tuple[str, str]:
        """Parse judge's response to extract winner and reasoning"""
        lines = judgment.split('\n')
        winner = None
        reasoning = ""
        
        for line in lines:
            if line.startswith("WINNER:"):
                winner_text = line.replace("WINNER:", "").strip()
                if "Government" in winner_text:
                    winner = "Government"
                elif "Opposition" in winner_text:
                    winner = "Opposition"
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        # Fallback parsing if format not followed
        if not winner:
            if "Government" in judgment and "Opposition" not in judgment:
                winner = "Government"
            elif "Opposition" in judgment and "Government" not in judgment:
                winner = "Opposition"
            else:
                # Default fallback - look for stronger language
                gov_count = judgment.lower().count("government")
                opp_count = judgment.lower().count("opposition")
                winner = "Government" if gov_count > opp_count else "Opposition"
        
        if not reasoning:
            reasoning = judgment
            
        return winner, reasoning