"""
Batch Execution System
Runs multiple debates systematically for statistical analysis.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os

import pandas as pd
from loguru import logger
from tqdm.asyncio import tqdm

from debate_engine import DebateEngine, SingleDebateConfig, DebateResult, DebateMetrics
from llm_personas import PersonaLibrary
from claude_client import ClaudeDebateClient


@dataclass 
class BatchConfig:
    debate_names: List[str]  # List of debate config names to run
    num_runs_per_debate: int = 10
    persona_pairs: Optional[List[tuple[str, str]]] = None  # If None, use all combinations per debate
    max_concurrent_debates: int = 3
    save_intermediate_results: bool = True
    output_dir: str = "debate_results"
    balanced_sampling: bool = False  # Enhanced sampling for statistical balance


@dataclass
class BatchResults:
    config: BatchConfig
    results: List[DebateResult]
    performance_report: Dict[str, Any]
    execution_time: float
    timestamp: str


class BatchDebateRunner:
    """Manages batch execution of debates for systematic analysis"""
    
    def __init__(self, debate_engine: DebateEngine):
        self.debate_engine = debate_engine
        self.persona_library = debate_engine.persona_library
        
    async def run_batch(self, config: BatchConfig) -> BatchResults:
        """Execute a batch of debates according to configuration"""
        
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Starting batch run: {len(config.debate_names)} debates, {config.num_runs_per_debate} runs each")
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Generate all debate configurations
        debate_configs = self._generate_debate_configs(config)
        logger.info(f"Total debates to run: {len(debate_configs)}")
        
        # Execute debates with concurrency control
        semaphore = asyncio.Semaphore(config.max_concurrent_debates)
        results = []
        
        async def run_single_debate(debate_config: SingleDebateConfig, batch_id: int) -> DebateResult:
            async with semaphore:
                try:
                    result = await self.debate_engine.run_debate(debate_config)
                    
                    # Save intermediate result if requested
                    if config.save_intermediate_results:
                        await self._save_intermediate_result(result, config.output_dir, batch_id, timestamp)
                    
                    return result
                except Exception as e:
                    logger.error(f"Batch debate {batch_id} failed: {e}")
                    # Return error result
                    return DebateResult(
                        debate=None,
                        winner="Error", 
                        confidence_score=0.0,
                        total_duration=0.0,
                        speech_durations=[],
                        error_occurred=True,
                        error_message=str(e)
                    )
        
        # Execute all debates with progress bar
        tasks = [
            run_single_debate(config, i) 
            for i, config in enumerate(debate_configs)
        ]
        
        results = await tqdm.gather(*tasks, desc="Running debates")
        
        execution_time = time.time() - start_time
        
        # Generate performance report
        performance_report = DebateMetrics.generate_performance_report(results)
        
        # Create batch results
        batch_results = BatchResults(
            config=config,
            results=results,
            performance_report=performance_report,
            execution_time=execution_time,
            timestamp=timestamp
        )
        
        # Save final results
        await self._save_batch_results(batch_results, config.output_dir)
        
        logger.success(f"Batch completed: {len(results)} debates in {execution_time:.1f}s")
        return batch_results
    
    def _generate_debate_configs(self, config: BatchConfig) -> List[SingleDebateConfig]:
        """Generate all debate configurations for the batch"""
        debate_configs = []
        
        for debate_name in config.debate_names:
            # Get persona pairs for this specific debate
            if config.persona_pairs:
                persona_pairs = config.persona_pairs
            else:
                persona_pairs = self.persona_library.get_persona_pairs_for_debate(debate_name)
            
            # Enhanced sampling for balanced statistical coverage
            if config.balanced_sampling:
                persona_pairs = self._apply_balanced_sampling(persona_pairs, debate_name)
            
            logger.info(f"Debate '{debate_name}': {len(persona_pairs)} persona combinations")
            
            for run_num in range(config.num_runs_per_debate):
                for gov_persona_id, opp_persona_id in persona_pairs:
                    # Alternate which persona is government vs opposition
                    if run_num % 2 == 0:
                        debate_config = SingleDebateConfig(
                            debate_name=debate_name,
                            government_persona_id=gov_persona_id,
                            opposition_persona_id=opp_persona_id
                        )
                    else:
                        debate_config = SingleDebateConfig(
                            debate_name=debate_name,
                            government_persona_id=opp_persona_id,
                            opposition_persona_id=gov_persona_id
                        )
                    
                    debate_configs.append(debate_config)
        
        return debate_configs
    
    def _apply_balanced_sampling(self, persona_pairs: List[tuple[str, str]], debate_name: str) -> List[tuple[str, str]]:
        """Apply statistical sampling adjustments for balanced coverage"""
        
        # Identify moderate/neutral personas that benefit from additional statistical coverage
        moderate_indicators = ['neutral', 'realist', 'pragmatist', 'analyst', 'strategist']
        
        enhanced_pairs = []
        base_pairs = []
        
        for p1, p2 in persona_pairs:
            # Check if this pair involves moderate personas
            p1_moderate = any(indicator in p1.lower() for indicator in moderate_indicators)
            p2_moderate = any(indicator in p2.lower() for indicator in moderate_indicators)
            
            if p1_moderate or p2_moderate:
                enhanced_pairs.append((p1, p2))
            else:
                base_pairs.append((p1, p2))
        
        # Return with enhanced sampling for moderate pairs (for statistical robustness)
        result = base_pairs.copy()
        result.extend(enhanced_pairs * 2)  # Additional samples for statistical significance
        
        return result
    
    async def _save_intermediate_result(
        self, 
        result: DebateResult, 
        output_dir: str, 
        batch_id: int,
        timestamp: str
    ):
        """Save individual debate result"""
        filename = f"{output_dir}/debate_{timestamp}_{batch_id:04d}.json"
        
        # Convert result to serializable format
        result_data = {
            "batch_id": batch_id,
            "timestamp": timestamp,
            "topic": result.debate.topic if result.debate else "Error",
            "government_persona_id": result.debate.government_side if result.debate else "Error",
            "opposition_persona_id": result.debate.opposition_side if result.debate else "Error",
            "winner": result.winner,
            "confidence_score": result.confidence_score,
            "total_duration": result.total_duration,
            "speech_durations": result.speech_durations,
            "error_occurred": result.error_occurred,
            "error_message": result.error_message,
            "speeches": [
                {
                    "speaker": speech.speaker.value,
                    "content": speech.content,
                    "word_count": speech.word_count,
                    "token_count": speech.token_count
                }
                for speech in (result.debate.speeches if result.debate else [])
            ],
            "judge_reasoning": result.debate.judge_reasoning if result.debate else None
        }
        
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    async def _save_batch_results(self, batch_results: BatchResults, output_dir: str):
        """Save complete batch results and analysis"""
        
        # Save summary report
        summary_file = f"{output_dir}/batch_summary_{batch_results.timestamp}.json"
        summary_data = {
            "timestamp": batch_results.timestamp,
            "config": {
                "debate_names": batch_results.config.debate_names,
                "num_runs_per_debate": batch_results.config.num_runs_per_debate,
                "max_concurrent_debates": batch_results.config.max_concurrent_debates
            },
            "execution_time": batch_results.execution_time,
            "performance_report": batch_results.performance_report
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save detailed CSV for analysis
        csv_file = f"{output_dir}/batch_results_{batch_results.timestamp}.csv"
        df_data = []
        
        for i, result in enumerate(batch_results.results):
            if result.debate:
                df_data.append({
                    "debate_id": i,
                    "topic": result.debate.topic,
                    "government_persona_id": result.debate.government_side,
                    "opposition_persona_id": result.debate.opposition_side,
                    "winner": result.winner,
                    "confidence_score": result.confidence_score,
                    "total_duration": result.total_duration,
                    "avg_speech_duration": sum(result.speech_durations) / len(result.speech_durations) if result.speech_durations else 0,
                    "total_words": sum(s.word_count for s in result.debate.speeches),
                    "error_occurred": result.error_occurred,
                    "error_message": result.error_message
                })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Batch results saved to {output_dir}/")


class TopicLibrary:
    """Manages debate topics for testing"""
    
    @staticmethod
    def get_default_topics() -> List[str]:
        """Get default set of debate topics for testing"""
        return [
            "This house believes that artificial intelligence will do more harm than good",
            "This house believes that social media platforms should be held legally responsible for user-generated content",
            "This house believes that governments should implement universal basic income"
        ]
    
    @staticmethod
    def get_political_topics() -> List[str]:
        """Political debate topics"""
        return [
            "This house believes that democratic governments should prioritize economic growth over environmental protection",
            "This house believes that wealthy nations have a moral obligation to accept unlimited refugees",
            "This house believes that voting should be mandatory in democratic elections"
        ]
    
    @staticmethod
    def get_technology_topics() -> List[str]:
        """Technology-focused debate topics"""
        return [
            "This house believes that cryptocurrency should be banned",
            "This house believes that tech companies should be broken up to prevent monopolies",
            "This house believes that gene editing should be allowed for human enhancement"
        ]
    
    @staticmethod
    def get_ethics_topics() -> List[str]:
        """Ethical debate topics"""
        return [
            "This house believes that animal testing should be completely banned",
            "This house believes that individuals have a right to be forgotten online", 
            "This house believes that parents should not be allowed to refuse medical treatment for their children"
        ]