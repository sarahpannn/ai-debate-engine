"""
Run additional debates with balanced sampling for better statistical coverage
"""

import asyncio
import os
from datetime import datetime

from claude_client import ClaudeDebateClient
from llm_personas import PersonaLibrary
from debate_engine import DebateEngine
from batch_runner import BatchDebateRunner, BatchConfig


async def run_balanced_batch():
    """Run debates with enhanced statistical sampling"""
    
    print("📊 RUNNING STATISTICALLY BALANCED DEBATE BATCH")
    print("=" * 60)
    
    # Setup
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return
    
    claude_client = ClaudeDebateClient(api_key=api_key)
    persona_library = PersonaLibrary(config_dir="configs")
    debate_engine = DebateEngine(claude_client, persona_library)
    batch_runner = BatchDebateRunner(debate_engine)
    
    # Get all available debates
    available_debates = persona_library.list_debates()
    
    # Configure batch with balanced sampling enabled
    config = BatchConfig(
        debate_names=available_debates,
        num_runs_per_debate=4,  # Additional runs for statistical robustness
        max_concurrent_debates=2,
        balanced_sampling=True, 
        output_dir=f"balanced_debates_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print(f"🎯 Running balanced batch across {len(config.debate_names)} debates")
    print(f"📈 Enhanced sampling enabled for improved statistical coverage")
    
    # Show what enhanced sampling will do
    for debate_name in available_debates:
        base_pairs = persona_library.get_persona_pairs_for_debate(debate_name)
        enhanced_pairs = batch_runner._apply_balanced_sampling(base_pairs, debate_name)
        debate_config = persona_library.get_debate_config(debate_name)
        
        print(f"   {debate_config.debate_info.title}:")
        print(f"     Base pairs: {len(base_pairs)}")
        print(f"     Enhanced pairs: {len(enhanced_pairs)} (+{len(enhanced_pairs) - len(base_pairs)} for balance)")
    
    # Run the balanced batch
    batch_results = await batch_runner.run_batch(config)
    
    # Display results
    report = batch_results.performance_report
    print(f"\n📈 BALANCED BATCH COMPLETED")
    print(f"Total debates: {report['total_debates']}")
    print(f"Successful: {report['successful_debates']}")
    print(f"Execution time: {batch_results.execution_time:.1f}s")
    
    print(f"\n🏆 PERFORMANCE DISTRIBUTION:")
    for persona, win_rate in sorted(report['persona_win_rates'].items(), key=lambda x: x[1], reverse=True):
        print(f"- {persona}: {win_rate:.1%}")
    
    print(f"\n💾 Results saved to: {config.output_dir}/")
    print(f"📊 Enhanced sampling provides better statistical coverage of moderate personas")
    
    return batch_results


if __name__ == "__main__":
    asyncio.run(run_balanced_batch())