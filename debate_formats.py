"""
Parliamentary Debate Format Handler
Implements the 6-speech Parliamentary debate structure with word/token limits.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import tiktoken


class SpeechType(Enum):
    PM = "Prime Minister"
    LO = "Leader of Opposition" 
    MG = "Member of Government"
    MO = "Member of Opposition"
    GW = "Government Whip"
    OW = "Opposition Whip"


@dataclass
class Speech:
    speaker: SpeechType
    content: str
    word_count: int
    token_count: int
    timestamp: float


@dataclass
class ParliamentaryDebate:
    topic: str
    government_side: str  # Gov persona
    opposition_side: str  # Opp persona
    speeches: List[Speech]
    winner: Optional[str] = None
    judge_reasoning: Optional[str] = None
    
    
class DebateFormat:
    def __init__(self, max_words: int = 500):
        self.max_words = max_words
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Parliamentary debate order
        self.speech_order = [
            SpeechType.PM,   # Government opens
            SpeechType.LO,   # Opposition responds
            SpeechType.MG,   # Government member
            SpeechType.MO,   # Opposition member
            SpeechType.OW,   # Opposition whip (summary)
            SpeechType.GW    # Government whip (summary)
        ]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def validate_speech(self, content: str) -> tuple[bool, str]:
        """Validate speech meets format requirements"""
        word_count = self.count_words(content)
        
        if word_count > self.max_words:
            return False, f"Speech exceeds {self.max_words} words ({word_count} words)"
        
        if word_count < 50:  # Minimum threshold
            return False, f"Speech too short ({word_count} words, minimum 50)"
            
        return True, "Valid"
    
    def create_speech(self, speaker: SpeechType, content: str, timestamp: float) -> Speech:
        """Create a validated Speech object"""
        is_valid, message = self.validate_speech(content)
        if not is_valid:
            raise ValueError(f"Invalid speech: {message}")
            
        return Speech(
            speaker=speaker,
            content=content,
            word_count=self.count_words(content),
            token_count=self.count_tokens(content),
            timestamp=timestamp
        )
    
    def is_government_speaker(self, speaker: SpeechType) -> bool:
        """Check if speaker is on government side"""
        return speaker in [SpeechType.PM, SpeechType.MG, SpeechType.GW]
    
    def get_next_speaker(self, current_speeches: List[Speech]) -> Optional[SpeechType]:
        """Get the next speaker based on parliamentary order"""
        if len(current_speeches) >= len(self.speech_order):
            return None  # Debate complete
            
        return self.speech_order[len(current_speeches)]
    
    def is_debate_complete(self, speeches: List[Speech]) -> bool:
        """Check if all 6 speeches have been delivered"""
        return len(speeches) == len(self.speech_order)