"""
Combined ELO Analysis for Original + Balanced Batch Results
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from elo_analyzer import DebateELOAnalyzer


class CombinedDebateAnalyzer(DebateELOAnalyzer):
    """Analyzer that combines multiple results directories"""
    
    def load_combined_results(self, results_dirs: list[str]) -> pd.DataFrame:
        """Load and combine debate results from multiple directories"""
        
        combined_data = []
        
        for results_dir in results_dirs:
            results_path = Path(results_dir)
            csv_files = list(results_path.glob("batch_results_*.csv"))
            
            if not csv_files:
                print(f"⚠️  No CSV found in {results_dir}")
                continue
            
            csv_file = csv_files[0]  # Use most recent
            df = pd.read_csv(csv_file)
            
            # Filter out error cases
            df = df[df['error_occurred'] == False].copy()
            
            # Add source directory info
            df['source_directory'] = results_dir
            
            combined_data.append(df)
            print(f"✅ Loaded {len(df)} successful debates from {csv_file}")
        
        if not combined_data:
            raise ValueError("No valid data found in any directory")
        
        # Combine all data
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"🎯 Total combined debates: {len(combined_df)}")
        
        return combined_df


def analyze_combined_debate_results(results_dirs: list[str], output_dir: str = "combined_analysis", use_confidence: bool = True):
    """Analyze combined results from multiple directories"""
    
    analyzer = CombinedDebateAnalyzer()
    
    # Load combined results
    print("📂 Loading combined results...")
    df = analyzer.load_combined_results(results_dirs)
    
    # Show breakdown by source
    print("\n📊 Results breakdown by source:")
    source_counts = df['source_directory'].value_counts()
    for source, count in source_counts.items():
        print(f"  - {source}: {count} debates")
    
    # Calculate ELO ratings on combined data
    print("\n🏆 Calculating combined ELO ratings...")
    elos = analyzer.calculate_elo_ratings(df, use_confidence=use_confidence)
    
    # Calculate bear-bull spectrum
    print("\n🐻🐂 Calculating bear-bull spectrum...")
    debate_topics = {topic: topic for topic in df['topic'].unique()}
    spectrum = analyzer.calculate_bear_bull_spectrum(df, debate_topics)
    
    # Generate report
    print("\n📈 Generating combined analysis report...")
    report = analyzer.generate_statistics_report()
    print(report)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create visualization
    print(f"\n📈 Creating combined bear-bull spectrum visualization...")
    confidence_suffix = "_with_confidence" if use_confidence else "_no_confidence"
    output_path = f"{output_dir}/combined_bear_bull_spectrum{confidence_suffix}.png"
    analyzer.create_bear_bull_visualization(output_path)
    
    # Save detailed results
    results = {
        'elo_ratings': elos,
        'bear_bull_scores': spectrum,
        'analysis_report': report,
        'used_confidence_weighting': use_confidence,
        'source_directories': results_dirs,
        'total_debates': len(df),
        'debates_by_source': source_counts.to_dict()
    }
    
    analysis_filename = f"combined_elo_analysis{'_with_confidence' if use_confidence else '_no_confidence'}.json"
    with open(f"{output_dir}/{analysis_filename}", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Combined analysis complete! Results saved to {output_dir}/")
    
    return analyzer


if __name__ == "__main__":
    # Analyze combined results
    results_directories = [
        "all_debates_results_20251013_173433",
        "balanced_debates_20251013_213432"
    ]
    
    print("🎯 COMBINED ANALYSIS WITH CONFIDENCE WEIGHTING:")
    print("=" * 60)
    analyzer_with_confidence = analyze_combined_debate_results(
        results_directories, 
        output_dir="combined_analysis", 
        use_confidence=True
    )
    
    print("\n" + "=" * 80)
    print("🎯 COMBINED ANALYSIS WITHOUT CONFIDENCE WEIGHTING:")
    print("=" * 60)
    analyzer_no_confidence = analyze_combined_debate_results(
        results_directories, 
        output_dir="combined_analysis", 
        use_confidence=False
    )
    
    print("\n" + "=" * 80)
    print("🔍 COMPARISON OF CONFIDENCE WEIGHTING:")
    print("=" * 60)
    
    # Compare the two approaches
    with_conf = analyzer_with_confidence.persona_elos
    without_conf = analyzer_no_confidence.persona_elos
    
    print("ELO Differences (No Confidence - With Confidence):")
    for persona in sorted(with_conf.keys()):
        diff = without_conf[persona] - with_conf[persona]
        direction = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
        print(f"{direction} {persona:25s}: {diff:+6.1f}")
    
    avg_diff = np.mean([abs(without_conf[p] - with_conf[p]) for p in with_conf.keys()])
    print(f"\nAverage absolute difference: {avg_diff:.1f} ELO points")