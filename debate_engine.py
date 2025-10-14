"""
Debate Execution Engine
Orchestrates full Parliamentary debates between LLM personas.
"""

import asyncio
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from loguru import logger
import pandas as pd

from debate_formats import ParliamentaryDebate, Speech, SpeechType, DebateFormat
from llm_personas import LLMPersona, PersonaLibrary
from claude_client import ClaudeDebateClient, DebateContext


@dataclass
class SingleDebateConfig:
    debate_name: str
    government_persona_id: str
    opposition_persona_id: str
    timeout_per_speech: int = 120  # seconds
    

@dataclass
class DebateResult:
    debate: ParliamentaryDebate
    winner: str
    confidence_score: float
    total_duration: float
    speech_durations: List[float]
    error_occurred: bool = False
    error_message: str = ""


class DebateEngine:
    """Core engine for running Parliamentary debates"""
    
    def __init__(self, claude_client: ClaudeDebateClient, persona_library: PersonaLibrary):
        self.claude_client = claude_client
        self.persona_library = persona_library
        self.debate_format = DebateFormat()
        
    async def run_debate(self, config: SingleDebateConfig) -> DebateResult:
        """Execute a complete Parliamentary debate"""
        
        start_time = time.time()
        speech_durations = []
        
        try:
            # Get debate configuration
            debate_config = self.persona_library.get_debate_config(config.debate_name)
            if not debate_config:
                raise ValueError(f"Debate config not found: {config.debate_name}")
            
            topic = debate_config.debate_info.topic
            logger.info(f"Starting debate: '{topic}'")
            logger.info(f"Government: {config.government_persona_id} vs Opposition: {config.opposition_persona_id}")
            
            # Get personas
            gov_persona = debate_config.personas.get(config.government_persona_id)
            opp_persona = debate_config.personas.get(config.opposition_persona_id)
            
            if not gov_persona or not opp_persona:
                raise ValueError(f"Invalid personas: {config.government_persona_id}, {config.opposition_persona_id}")
            
            # Initialize debate
            debate = ParliamentaryDebate(
                topic=topic,
                government_side=config.government_persona_id,
                opposition_side=config.opposition_persona_id,
                speeches=[]
            )
            
            # Execute all 6 speeches in order
            for speech_num in range(6):
                speech_type = self.debate_format.speech_order[speech_num]
                is_government = self.debate_format.is_government_speaker(speech_type)
                
                # Select appropriate persona and side
                current_persona = gov_persona if is_government else opp_persona
                side = "government" if is_government else "opposition"
                
                logger.info(f"Generating speech {speech_num + 1}/6: {speech_type.value} ({side})")
                
                # Build context for this speech
                context = DebateContext(
                    topic=topic,
                    current_speeches=debate.speeches.copy(),
                    persona=current_persona,
                    side=side
                )
                
                # Generate speech with timeout
                speech_start = time.time()
                try:
                    speech_content = await asyncio.wait_for(
                        self.claude_client.generate_speech(context, speech_type),
                        timeout=config.timeout_per_speech
                    )
                    
                    speech_duration = time.time() - speech_start
                    speech_durations.append(speech_duration)
                    
                    # Create speech object
                    speech = self.debate_format.create_speech(
                        speaker=speech_type,
                        content=speech_content,
                        timestamp=speech_start
                    )
                    
                    debate.speeches.append(speech)
                    
                    logger.success(f"Speech {speech_num + 1} completed ({speech.word_count} words, {speech_duration:.1f}s)")
                    
                except asyncio.TimeoutError:
                    raise RuntimeError(f"Speech {speech_num + 1} timed out after {config.timeout_per_speech}s")
                
                # Brief pause between speeches
                await asyncio.sleep(1)
            
            # Judge the debate
            logger.info("Judging debate...")
            judge_start = time.time()
            
            winner, reasoning = await asyncio.wait_for(
                self.claude_client.judge_debate(topic, debate.speeches),
                timeout=config.timeout_per_speech
            )
            
            judge_duration = time.time() - judge_start
            
            debate.winner = winner
            debate.judge_reasoning = reasoning
            
            total_duration = time.time() - start_time
            
            logger.success(f"Debate completed! Winner: {winner} (Total time: {total_duration:.1f}s)")
            
            # Calculate confidence score based on reasoning length and decisiveness
            confidence_score = self._calculate_confidence_score(reasoning)
            
            return DebateResult(
                debate=debate,
                winner=winner,
                confidence_score=confidence_score,
                total_duration=total_duration,
                speech_durations=speech_durations
            )
            
        except Exception as e:
            error_msg = f"Debate failed: {str(e)}"
            logger.error(error_msg)
            
            # Return partial result with error
            debate_config = self.persona_library.get_debate_config(config.debate_name)
            topic = debate_config.debate_info.topic if debate_config else "Unknown"
            
            return DebateResult(
                debate=ParliamentaryDebate(
                    topic=topic,
                    government_side=config.government_persona_id,
                    opposition_side=config.opposition_persona_id,
                    speeches=debate.speeches if 'debate' in locals() else []
                ),
                winner="Error",
                confidence_score=0.0,
                total_duration=time.time() - start_time,
                speech_durations=speech_durations,
                error_occurred=True,
                error_message=error_msg
            )
    
    def _calculate_confidence_score(self, reasoning: str) -> float:
        """Calculate judge confidence score from reasoning text"""
        
        # Factors that indicate high confidence
        high_confidence_words = [
            "clearly", "obviously", "decisively", "overwhelming", "strong", 
            "convincing", "superior", "dominant", "excellent", "compelling"
        ]
        
        # Factors that indicate low confidence  
        low_confidence_words = [
            "close", "narrow", "difficult", "marginal", "slight", "barely",
            "somewhat", "perhaps", "possibly", "uncertain", "unclear"
        ]
        
        reasoning_lower = reasoning.lower()
        
        high_count = sum(1 for word in high_confidence_words if word in reasoning_lower)
        low_count = sum(1 for word in low_confidence_words if word in reasoning_lower)
        
        # Base confidence from word analysis
        word_confidence = min(1.0, max(0.3, 0.5 + (high_count - low_count) * 0.1))
        
        # Length factor (longer reasoning might indicate more thorough analysis)
        length_factor = min(1.0, len(reasoning.split()) / 100)  # Normalize to ~100 words
        
        # Combine factors
        final_confidence = (word_confidence * 0.7) + (length_factor * 0.3)
        
        return round(final_confidence, 3)
    
    async def validate_setup(self) -> bool:
        """Validate that the debate engine is properly configured"""
        try:
            # Test basic persona access
            personas = self.persona_library.list_personas()
            if len(personas) < 2:
                logger.error("Need at least 2 personas for debates")
                return False
            
            # Test Claude client with a simple request
            test_config = DebateConfig(
                topic="Test topic",
                government_persona=personas[0],
                opposition_persona=personas[1],
                max_words_per_speech=50
            )
            
            logger.info("Validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


class DebateMetrics:
    """Calculate performance metrics from debate results"""
    
    @staticmethod
    def calculate_win_rate(results: List[DebateResult], persona_name: str) -> float:
        """Calculate win rate for a specific persona"""
        if not results:
            return 0.0
            
        relevant_results = [
            r for r in results 
            if not r.error_occurred and (
                r.debate.government_side == persona_name or 
                r.debate.opposition_side == persona_name
            )
        ]
        
        if not relevant_results:
            return 0.0
        
        wins = sum(1 for r in relevant_results if (
            (r.debate.government_side == persona_name and r.winner == "Government") or
            (r.debate.opposition_side == persona_name and r.winner == "Opposition")
        ))
        
        return wins / len(relevant_results)
    
    @staticmethod
    def calculate_side_bias(results: List[DebateResult]) -> Dict[str, float]:
        """Calculate if there's bias toward government or opposition"""
        if not results:
            return {"government": 0.5, "opposition": 0.5}
        
        valid_results = [r for r in results if not r.error_occurred]
        if not valid_results:
            return {"government": 0.5, "opposition": 0.5}
        
        gov_wins = sum(1 for r in valid_results if r.winner == "Government")
        total = len(valid_results)
        
        return {
            "government": gov_wins / total,
            "opposition": (total - gov_wins) / total
        }
    
    @staticmethod
    def generate_performance_report(results: List[DebateResult]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not results:
            return {"error": "No results to analyze"}
        
        valid_results = [r for r in results if not r.error_occurred]
        error_rate = (len(results) - len(valid_results)) / len(results)
        
        # Get all unique personas
        personas = set()
        for r in valid_results:
            personas.add(r.debate.government_side)
            personas.add(r.debate.opposition_side)
        
        # Calculate win rates
        win_rates = {}
        for persona in personas:
            win_rates[persona] = DebateMetrics.calculate_win_rate(valid_results, persona)
        
        # Calculate average metrics
        avg_duration = sum(r.total_duration for r in valid_results) / len(valid_results) if valid_results else 0
        avg_confidence = sum(r.confidence_score for r in valid_results) / len(valid_results) if valid_results else 0
        
        # Side bias
        side_bias = DebateMetrics.calculate_side_bias(valid_results)
        
        return {
            "total_debates": len(results),
            "successful_debates": len(valid_results),
            "error_rate": error_rate,
            "average_duration_seconds": avg_duration,
            "average_judge_confidence": avg_confidence,
            "side_bias": side_bias,
            "persona_win_rates": win_rates,
            "top_performer": max(win_rates.items(), key=lambda x: x[1]) if win_rates else None
        }