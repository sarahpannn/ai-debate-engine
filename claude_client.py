"""
Claude API Client for Debate Simulation
Handles communication with Claude API for debate participants and judge.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

import anthropic
from anthropic import Anthropic
from loguru import logger

from llm_personas import LLMPersona
from debate_formats import SpeechType, Speech, DebateFormat


@dataclass
class DebateContext:
    topic: str
    current_speeches: list[Speech]
    persona: LLMPersona
    side: str  # "government" or "opposition"


class ClaudeDebateClient:
    """Claude API client optimized for debate simulation"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.debate_format = DebateFormat()
        
    async def generate_speech(
        self, 
        context: DebateContext,
        speech_type: SpeechType,
        max_retries: int = 3
    ) -> str:
        """Generate a debate speech using Claude API"""
        
        # Build context from previous speeches
        speech_history = self._format_speech_history(context.current_speeches)
        
        # Create debate-specific prompt
        prompt = self._build_speech_prompt(
            topic=context.topic,
            persona=context.persona,
            speech_type=speech_type,
            side=context.side,
            speech_history=speech_history
        )
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating {speech_type.value} speech (attempt {attempt + 1})")
                
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=context.persona.max_tokens,
                    temperature=context.persona.temperature,
                    system=context.persona.system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                speech_content = response.content[0].text.strip()
                
                # Validate speech meets format requirements
                is_valid, message = self.debate_format.validate_speech(speech_content)
                if is_valid:
                    logger.success(f"Generated valid {speech_type.value} speech")
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
    
    async def judge_debate(
        self, 
        topic: str, 
        speeches: list[Speech],
        max_retries: int = 3
    ) -> tuple[str, str]:
        """Use Claude as judge to determine debate winner"""
        
        judge_prompt = self._build_judge_prompt(topic, speeches)
        
        judge_system = """You are an expert debate judge evaluating a Parliamentary debate.
        
        Evaluation Criteria:
        1. Argument Quality: Logic, evidence, reasoning
        2. Refutation: How well each side addressed opponent's points
        3. Structure: Organization and flow of arguments
        4. Persuasiveness: Overall convincing power
        
        You must declare either "Government" or "Opposition" as the winner.
        Provide clear reasoning for your decision.
        
        Format your response as:
        WINNER: [Government/Opposition]
        REASONING: [Your detailed analysis]"""
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Judging debate (attempt {attempt + 1})")
                
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.3,  # Lower temperature for more consistent judging
                    system=judge_system,
                    messages=[{"role": "user", "content": judge_prompt}]
                )
                
                judgment = response.content[0].text.strip()
                winner, reasoning = self._parse_judgment(judgment)
                
                logger.success(f"Debate judged: {winner} wins")
                return winner, reasoning
                
            except Exception as e:
                logger.error(f"Error judging debate (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise RuntimeError(f"Failed to judge debate after {max_retries} attempts")
    
    def _format_speech_history(self, speeches: list[Speech]) -> str:
        """Format previous speeches for context"""
        if not speeches:
            return "This is the opening speech of the debate."
        
        history = "Previous speeches in this debate:\n\n"
        for i, speech in enumerate(speeches, 1):
            side = "Government" if self.debate_format.is_government_speaker(speech.speaker) else "Opposition"
            history += f"Speech {i} - {speech.speaker.value} ({side}):\n"
            history += f"{speech.content}\n\n"
        
        return history
    
    def _build_speech_prompt(
        self,
        topic: str,
        persona: LLMPersona, 
        speech_type: SpeechType,
        side: str,
        speech_history: str
    ) -> str:
        """Build prompt for speech generation"""
        
        role_descriptions = {
            SpeechType.PM: "You are the Prime Minister opening for the Government. Set up your case clearly.",
            SpeechType.LO: "You are the Leader of Opposition. Respond to Government's case and present your alternative.",
            SpeechType.MG: "You are a Member of Government. Extend and strengthen your side's arguments.",
            SpeechType.MO: "You are a Member of Opposition. Attack Government's case and build Opposition's position.",
            SpeechType.OW: "You are Opposition Whip. Summarize why Opposition should win this debate.",
            SpeechType.GW: "You are Government Whip. Summarize why Government should win this debate."
        }
        
        prompt = f"""You are participating in a Parliamentary debate on the topic:
        
        "{topic}"
        
        {role_descriptions[speech_type]}
        
        Your side: {side.title()}
        
        {speech_history}
        
        Requirements:
        - Maximum {self.debate_format.max_words} words
        - Address the topic directly
        - If responding to previous speeches, engage with their arguments
        - Use Parliamentary debate conventions
        
        Deliver your speech now:"""
        
        return prompt
    
    def _build_judge_prompt(self, topic: str, speeches: list[Speech]) -> str:
        """Build prompt for debate judging"""
        
        prompt = f"""Please judge this Parliamentary debate on the topic: "{topic}"
        
        Here are all six speeches in order:
        
        """
        
        for i, speech in enumerate(speeches, 1):
            side = "Government" if self.debate_format.is_government_speaker(speech.speaker) else "Opposition"
            prompt += f"Speech {i} - {speech.speaker.value} ({side}):\n"
            prompt += f"{speech.content}\n\n"
        
        prompt += """
        Evaluate the debate based on:
        1. Strength of arguments and evidence
        2. Effective refutation of opponent's points  
        3. Logical structure and flow
        4. Overall persuasiveness
        
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