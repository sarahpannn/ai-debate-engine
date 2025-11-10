"""
Continuous Debate Runner with Elo Rating System
Runs debates until Claude credits are exhausted, tracking persona performance.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import traceback

from ai_service import ClaudeDebateService, DebateContext
from llm_personas import PersonaLibrary
from debate_engine import SingleDebateConfig
from debate_formats import SpeechType, ParliamentaryDebate, Speech
from elo_rating_system import EloRatingSystem, EloRating
import tiktoken


class DebateResult:
    """Store results of a single debate"""
    
    def __init__(self, config: SingleDebateConfig, topic: str, speeches: List[Speech], 
                 winner: str, reasoning: str, duration: float, 
                 gov_rating: Optional[EloRating] = None, opp_rating: Optional[EloRating] = None,
                 error: Optional[str] = None):
        self.config = config
        self.topic = topic
        self.speeches = speeches
        self.winner = winner
        self.reasoning = reasoning
        self.duration = duration
        self.gov_rating = gov_rating
        self.opp_rating = opp_rating
        self.error = error
        self.timestamp = datetime.now().isoformat()
        
        # Calculate stats
        self.total_words = sum(s.word_count for s in speeches)
        self.total_tokens = sum(s.token_count for s in speeches)
        self.government_speeches = len([s for i, s in enumerate(speeches) if i % 2 == 0])
        self.opposition_speeches = len([s for i, s in enumerate(speeches) if i % 2 == 1])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            "timestamp": self.timestamp,
            "debate_name": self.config.debate_name,
            "government_persona": self.config.government_persona_id,
            "opposition_persona": self.config.opposition_persona_id,
            "topic": self.topic,
            "winner": self.winner,
            "reasoning": self.reasoning[:500] + "..." if len(self.reasoning) > 500 else self.reasoning,
            "duration_seconds": self.duration,
            "total_words": self.total_words,
            "total_tokens": self.total_tokens,
            "government_speeches": self.government_speeches,
            "opposition_speeches": self.opposition_speeches,
            "speeches": [
                {
                    "speaker": speech.speaker.value,
                    "word_count": speech.word_count,
                    "token_count": speech.token_count,
                    "content_preview": speech.content[:200] + "..." if len(speech.content) > 200 else speech.content
                }
                for speech in self.speeches
            ],
            "error": self.error
        }
        
        # Add Elo ratings if available
        if self.gov_rating:
            result["government_rating"] = self.gov_rating.to_dict()
        if self.opp_rating:
            result["opposition_rating"] = self.opp_rating.to_dict()
        
        return result


class ContinuousDebateRunner:
    """Runs continuous debates with RAG integration and Elo tracking until credits exhausted"""
    
    def __init__(self, output_dir: str = "debate_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.service = ClaudeDebateService()
        self.persona_library = PersonaLibrary()
        self.elo_system = EloRatingSystem()
        
        # Load existing Elo ratings
        self.elo_file = self.output_dir / "elo_ratings.json"
        self.elo_system.load_ratings(self.elo_file)
        
        # Stats tracking
        self.debates_completed = 0
        self.total_words_generated = 0
        self.total_tokens_used = 0
        self.start_time = time.time()
        self.results: List[DebateResult] = []
        
        # Session file for persistence
        self.session_file = self.output_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        print(f"🎯 Continuous Debate Runner with Elo Rating System initialized")
        print(f"📁 Results will be saved to: {self.output_dir}")
        print(f"📊 Session file: {self.session_file}")
        print(f"🏆 Elo ratings file: {self.elo_file}")
    
    async def generate_all_debate_configs(self) -> List[SingleDebateConfig]:
        """Generate all possible debate configurations"""
        configs = []
        
        # Get all debate types and persona combinations
        debate_names = self.persona_library.list_debates()
        
        for debate_name in debate_names:
            personas = self.persona_library.list_personas_for_debate(debate_name)
            
            # Create all possible pairings
            for i, gov_persona in enumerate(personas):
                for opp_persona in personas[i+1:]:
                    # Create both orderings (A vs B, B vs A)
                    configs.append(SingleDebateConfig(
                        debate_name=debate_name,
                        government_persona_id=gov_persona,
                        opposition_persona_id=opp_persona,
                        timeout_per_speech=180
                    ))
                    
                    configs.append(SingleDebateConfig(
                        debate_name=debate_name,
                        government_persona_id=opp_persona,
                        opposition_persona_id=gov_persona,
                        timeout_per_speech=180
                    ))
        
        print(f"📋 Generated {len(configs)} debate configurations")
        return configs
    
    async def run_single_debate(self, config: SingleDebateConfig, debate_id: str = "") -> DebateResult:
        """Run a single debate and return results with updated Elo ratings"""
        start_time = time.time()
        
        try:
            # Get current Elo ratings before the debate
            gov_rating_before = self.elo_system.get_rating(config.debate_name, config.government_persona_id)
            opp_rating_before = self.elo_system.get_rating(config.debate_name, config.opposition_persona_id)
            
            print(f"  [{debate_id}] 🎪 Starting Debate: {config.debate_name}")
            print(f"  [{debate_id}] 🏛️ Gov: {config.government_persona_id} (Elo: {gov_rating_before.rating:.0f}) vs Opp: {config.opposition_persona_id} (Elo: {opp_rating_before.rating:.0f})")
            
            # Calculate expected winner
            expected_gov, expected_opp = self.elo_system.calculate_expected_score(
                gov_rating_before.rating, opp_rating_before.rating
            )
            favorite = "Government" if expected_gov > expected_opp else "Opposition"
            print(f"  [{debate_id}] 🎲 Expected winner: {favorite} ({max(expected_gov, expected_opp)*100:.1f}% chance)")
            
            # Get debate configuration and topic
            debate_config = self.persona_library.get_debate_config(config.debate_name)
            topic = debate_config.debate_info.topic
            
            # Get personas
            gov_persona = debate_config.personas.get(config.government_persona_id)
            opp_persona = debate_config.personas.get(config.opposition_persona_id)
            
            if not gov_persona or not opp_persona:
                raise ValueError(f"Missing personas: {config.government_persona_id}, {config.opposition_persona_id}")
            
            # Initialize debate
            debate = ParliamentaryDebate(
                topic=topic,
                government_side=config.government_persona_id,
                opposition_side=config.opposition_persona_id,
                speeches=[]
            )
            
            # Speech order: PM, LO, MG, MO, GW, OW
            speech_order = [
                (SpeechType.PM, True, gov_persona),   # Prime Minister (Government)
                (SpeechType.LO, False, opp_persona),  # Leader of Opposition
                (SpeechType.MG, True, gov_persona),   # Member of Government 
                (SpeechType.MO, False, opp_persona),  # Member of Opposition
                (SpeechType.GW, True, gov_persona),   # Government Whip
                (SpeechType.OW, False, opp_persona)   # Opposition Whip
            ]
            
            print(f"  [{debate_id}] 🎤 Generating all 6 speeches...")
            
            # Generate all speeches
            for i, (speech_type, is_government, persona) in enumerate(speech_order):
                side_name = "Government" if is_government else "Opposition"
                print(f"  [{debate_id}] {i+1}/6: Generating {speech_type.value} ({side_name})...")
                
                context = DebateContext(
                    topic=topic,
                    current_speeches=debate.speeches.copy(),
                    persona=persona,
                    side="government" if is_government else "opposition"
                )
                
                try:
                    speech_content = await self.service.generate_speech(context, speech_type)
                    
                    # Create speech object with token count
                    word_count = len(speech_content.split())
                    
                    # Calculate token count using tiktoken
                    encoding = tiktoken.get_encoding("cl100k_base")
                    token_count = len(encoding.encode(speech_content))
                    
                    speech = Speech(
                        speaker=speech_type,
                        content=speech_content,
                        word_count=word_count,
                        token_count=token_count,
                        timestamp=time.time() - start_time
                    )
                    
                    debate.speeches.append(speech)
                    
                    print(f"  [{debate_id}] ✓ {speech_type.value} completed ({speech.word_count} words, {token_count} tokens)")
                    
                except Exception as e:
                    print(f"  [{debate_id}] ✗ Failed to generate {speech_type.value}: {e}")
                    raise
            
            print(f"  [{debate_id}] ⚖️ Judging debate...")
            
            # Judge the debate
            winner, reasoning = await self.service.judge_debate(topic, debate.speeches)
            
            # Update Elo ratings based on result
            gov_rating_after, opp_rating_after = self.elo_system.update_ratings(
                config.debate_name, config.government_persona_id, config.opposition_persona_id, winner
            )
            
            duration = time.time() - start_time
            
            # Create result object with updated ratings
            result = DebateResult(
                config=config,
                topic=topic,
                speeches=debate.speeches,
                winner=winner,
                reasoning=reasoning,
                duration=duration,
                gov_rating=gov_rating_after,
                opp_rating=opp_rating_after
            )
            
            # Calculate rating changes
            gov_change = gov_rating_after.rating - gov_rating_before.rating
            opp_change = opp_rating_after.rating - opp_rating_before.rating
            
            print(f"  [{debate_id}] 🏆 DEBATE COMPLETE!")
            print(f"  [{debate_id}] Winner: {winner} | Duration: {duration:.1f}s | Words: {result.total_words} | Tokens: {result.total_tokens}")
            print(f"  [{debate_id}] 🏛️ Gov Elo: {gov_rating_before.rating:.0f} → {gov_rating_after.rating:.0f} ({gov_change:+.0f})")
            print(f"  [{debate_id}] 🗣️ Opp Elo: {opp_rating_before.rating:.0f} → {opp_rating_after.rating:.0f} ({opp_change:+.0f})")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Error in debate: {str(e)}"
            print(f"  [{debate_id}] ❌ DEBATE FAILED: {error_msg}")
            traceback.print_exc()
            
            # Return partial result with error
            return DebateResult(
                config=config,
                topic=getattr(debate_config, 'debate_info', {}).get('topic', 'Unknown topic'),
                speeches=[],
                winner="Error",
                reasoning=error_msg,
                duration=duration,
                error=error_msg
            )
    
    def save_session_state(self):
        """Save current session state and Elo ratings to disk"""
        session_data = {
            "start_time": self.start_time,
            "debates_completed": self.debates_completed,
            "total_words_generated": self.total_words_generated,
            "total_tokens_used": self.total_tokens_used,
            "runtime_seconds": time.time() - self.start_time,
            "results": [result.to_dict() for result in self.results]
        }
        
        # Save session data
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Save Elo ratings
        self.elo_system.save_ratings(self.elo_file)
        
        print(f"💾 Session saved to {self.session_file}")
        print(f"🏆 Elo ratings saved to {self.elo_file}")
    
    def print_session_stats(self):
        """Print current session statistics including Elo leaderboards"""
        runtime = time.time() - self.start_time
        
        print(f"\n📊 SESSION STATISTICS")
        print(f"{'='*50}")
        print(f"Runtime: {runtime/60:.1f} minutes")
        print(f"Debates completed: {self.debates_completed}")
        print(f"Total words generated: {self.total_words_generated:,}")
        print(f"Total tokens used: {self.total_tokens_used:,}")
        
        if self.debates_completed > 0:
            print(f"Average words per debate: {self.total_words_generated/self.debates_completed:.0f}")
            print(f"Average tokens per debate: {self.total_tokens_used/self.debates_completed:.0f}")
            print(f"Average time per debate: {runtime/self.debates_completed:.1f}s")
        
        # Winner statistics
        winners = [r.winner for r in self.results if r.winner not in ["Error"]]
        if winners:
            gov_wins = winners.count("Government")
            opp_wins = winners.count("Opposition")
            print(f"Government wins: {gov_wins} ({gov_wins/len(winners)*100:.1f}%)")
            print(f"Opposition wins: {opp_wins} ({opp_wins/len(winners)*100:.1f}%)")
        
        print(f"{'='*50}")
        
        # Print Elo leaderboards for each debate type
        debate_names = self.persona_library.list_debates()
        for debate_name in debate_names:
            self.elo_system.print_leaderboard(debate_name, limit=10)
        
        # Print overall leaderboard
        self.elo_system.print_leaderboard(None, limit=15)
        
        # Print rating distribution
        distribution = self.elo_system.get_rating_distribution()
        print(f"\n🎯 RATING DISTRIBUTION")
        print("=" * 25)
        for tier, count in distribution.items():
            if count > 0:
                print(f"{tier}: {count}")
    

    async def run_continuous_debates(self, max_debates: Optional[int] = None, 
                                   save_interval: int = 1):
        """Run debates continuously (sequential) until credits exhausted"""
        
        print(f"🚀 Starting continuous debate runner with Elo tracking...")
        print(f"💳 Will run until Claude credits exhausted or {max_debates or 'unlimited'} debates")
        print(f"💾 Saving session state every {save_interval} debates")
        print(f"🎲 Running debates sequentially with random sampling")
        
        # Generate all possible configurations
        all_configs = await self.generate_all_debate_configs()
        
        print(f"📋 Total configurations: {len(all_configs)}")
        print(f"🎲 Will randomly sample from configs for variety")
        
        # Main loop with random sampling
        import random
        
        while True:
            try:
                # Check if we've reached max debates
                if max_debates and self.debates_completed >= max_debates:
                    print(f"🎯 Reached maximum debates limit: {max_debates}")
                    break
                
                # Randomly sample next config
                config = random.choice(all_configs)
                
                print(f"\n🎪 DEBATE #{self.debates_completed + 1}")
                print(f"📖 {config.debate_name} | 🏛️ {config.government_persona_id} vs 🗣️ {config.opposition_persona_id}")
                
                # Run single debate
                result = await self.run_single_debate(config, f"D{self.debates_completed + 1}")
                
                # Update statistics
                self.results.append(result)
                self.debates_completed += 1
                self.total_words_generated += result.total_words
                self.total_tokens_used += result.total_tokens
                
                print(f"📊 PROGRESS: {self.debates_completed} debates completed")
                
                # Save session state periodically
                if self.debates_completed % save_interval == 0:
                    self.save_session_state()
                    self.print_session_stats()
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(2)
                
            except Exception as e:
                # Check if this might be a credits exhaustion error
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['credit', 'quota', 'limit', 'billing']):
                    print(f"💳 Credits likely exhausted: {e}")
                    break
                elif 'rate limit' in error_msg or '429' in error_msg:
                    print(f"⏳ Rate limited, waiting 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                else:
                    print(f"❌ Unexpected error: {e}")
                    traceback.print_exc()
                    
                    # Continue with next batch after short delay
                    await asyncio.sleep(10)
                    continue
        
        # Final save and statistics
        self.save_session_state()
        self.print_session_stats()
        
        print(f"\n🏁 CONTINUOUS DEBATE RUNNER COMPLETED!")
        print(f"📁 Final results saved to: {self.session_file}")
        print(f"🏆 Final Elo ratings saved to: {self.elo_file}")
        print(f"🎪 Total debates completed: {self.debates_completed}")
        
        # Print final distribution analysis
        debate_counts = {}
        persona_counts = {}
        
        for result in self.results:
            if result.error:
                continue
                
            debate_name = result.config.debate_name
            gov_persona = result.config.government_persona_id
            opp_persona = result.config.opposition_persona_id
            
            debate_counts[debate_name] = debate_counts.get(debate_name, 0) + 1
            persona_counts[gov_persona] = persona_counts.get(gov_persona, 0) + 1
            persona_counts[opp_persona] = persona_counts.get(opp_persona, 0) + 1
        
        print(f"\n📊 FINAL DISTRIBUTION:")
        print("Debates by type:")
        for debate, count in sorted(debate_counts.items()):
            print(f"  {debate}: {count}")
        print("Persona appearances:")
        for persona, count in sorted(persona_counts.items()):
            print(f"  {persona}: {count}")


async def main():
    """Main entry point for continuous debate runner"""
    
    # Check environment variables
    required_vars = ["ANTHROPIC_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY", "GEMINI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return
    
    print("✅ All required environment variables present")
    
    # Initialize and run
    runner = ContinuousDebateRunner()
    
    try:
        # Run until credits exhausted (or max 1000 debates as safety)
        await runner.run_continuous_debates(
            max_debates=1000, 
            save_interval=1
        )
        
    except KeyboardInterrupt:
        print(f"\n⏹️ User interrupted - saving final state...")
        runner.save_session_state()
        runner.print_session_stats()
    
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        traceback.print_exc()
        runner.save_session_state()


if __name__ == "__main__":
    asyncio.run(main())