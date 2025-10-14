"""
LLM Persona Management System
Loads debate-specific personas from configuration files.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class DebateInfo:
    title: str
    topic: str
    description: str
    max_words_per_speech: int
    max_tokens_per_speech: int
    temperature: float


@dataclass
class LLMPersona:
    name: str
    description: str
    system_prompt: str
    temperature: float = 0.8
    max_tokens: int = 700  # Conservative for 500 words + buffer
    
    
@dataclass
class DebateConfig:
    debate_info: DebateInfo
    personas: Dict[str, LLMPersona]


class PersonaLibrary:
    """Manages debate-specific personas loaded from config files"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.debate_configs: Dict[str, DebateConfig] = {}
        self.load_all_configs()
    
    def load_all_configs(self):
        """Load all debate configurations from config directory"""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        
        for config_file in self.config_dir.glob("*.json"):
            try:
                debate_config = self.load_debate_config(config_file)
                config_name = config_file.stem  # filename without extension
                self.debate_configs[config_name] = debate_config
                print(f"✅ Loaded debate config: {config_name}")
            except Exception as e:
                print(f"❌ Failed to load {config_file}: {e}")
    
    def load_debate_config(self, config_path: Path) -> DebateConfig:
        """Load a single debate configuration from JSON file"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Parse debate info
        debate_info_data = data["debate_info"]
        debate_info = DebateInfo(
            title=debate_info_data["title"],
            topic=debate_info_data["topic"],
            description=debate_info_data["description"],
            max_words_per_speech=debate_info_data["max_words_per_speech"],
            max_tokens_per_speech=debate_info_data["max_tokens_per_speech"],
            temperature=debate_info_data["temperature"]
        )
        
        # Parse personas
        personas = {}
        for persona_id, persona_data in data["personas"].items():
            persona = LLMPersona(
                name=persona_data["name"],
                description=persona_data["description"],
                system_prompt=persona_data["system_prompt"],
                temperature=debate_info.temperature,  # Use consistent temperature
                max_tokens=debate_info.max_tokens_per_speech
            )
            personas[persona_id] = persona
        
        return DebateConfig(debate_info=debate_info, personas=personas)
    
    def get_debate_config(self, debate_name: str) -> Optional[DebateConfig]:
        """Get a specific debate configuration"""
        return self.debate_configs.get(debate_name)
    
    def list_debates(self) -> List[str]:
        """Get list of available debate configurations"""
        return list(self.debate_configs.keys())
    
    def get_persona_from_debate(self, debate_name: str, persona_id: str) -> Optional[LLMPersona]:
        """Get a specific persona from a specific debate"""
        debate_config = self.get_debate_config(debate_name)
        if debate_config:
            return debate_config.personas.get(persona_id)
        return None
    
    def list_personas_for_debate(self, debate_name: str) -> List[str]:
        """Get list of persona IDs for a specific debate"""
        debate_config = self.get_debate_config(debate_name)
        if debate_config:
            return list(debate_config.personas.keys())
        return []
    
    def get_persona_pairs_for_debate(self, debate_name: str) -> List[tuple[str, str]]:
        """Get all possible pairs of personas for a specific debate"""
        personas = self.list_personas_for_debate(debate_name)
        pairs = []
        for i, p1 in enumerate(personas):
            for p2 in personas[i+1:]:
                pairs.append((p1, p2))
        return pairs
    
    def get_all_persona_pairs(self) -> List[tuple[str, str, str]]:
        """Get all possible pairs across all debates (debate_name, persona1, persona2)"""
        all_pairs = []
        for debate_name in self.list_debates():
            pairs = self.get_persona_pairs_for_debate(debate_name)
            for p1, p2 in pairs:
                all_pairs.append((debate_name, p1, p2))
        return all_pairs