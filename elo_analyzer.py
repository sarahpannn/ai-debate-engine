"""
ELO Rating System and Bear-Bull Spectrum Analysis for Debate Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from pathlib import Path


class DebateELOAnalyzer:
    """Analyze debate results using ELO rating system and bear-bull spectrum"""
    
    def __init__(self, k_factor: float = 32):
        self.k_factor = k_factor  # ELO sensitivity factor
        self.persona_elos = {}
        self.persona_bear_bull_scores = {}
        
    def load_results(self, results_dir: str) -> pd.DataFrame:
        """Load debate results from CSV file"""
        results_path = Path(results_dir)
        csv_files = list(results_path.glob("batch_results_*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No batch results CSV found in {results_dir}")
        
        csv_file = csv_files[0]  # Use most recent
        df = pd.read_csv(csv_file)
        
        # Filter out error cases
        df = df[df['error_occurred'] == False].copy()
        
        print(f"Loaded {len(df)} successful debates from {csv_file}")
        return df
    
    def calculate_elo_ratings(self, df: pd.DataFrame, use_confidence: bool = True) -> Dict[str, float]:
        """Calculate ELO ratings for all personas"""
        
        # Initialize all personas with starting ELO of 1500
        all_personas = set(df['government_persona_id'].unique()) | set(df['opposition_persona_id'].unique())
        self.persona_elos = {persona: 1500.0 for persona in all_personas}
        
        confidence_mode = "with confidence weighting" if use_confidence else "without confidence weighting"
        print(f"Calculating ELO for {len(all_personas)} personas ({confidence_mode}):")
        for persona in sorted(all_personas):
            print(f"  - {persona}")
        
        # Process each debate chronologically
        for _, row in df.iterrows():
            gov_persona = row['government_persona_id']
            opp_persona = row['opposition_persona_id']
            winner = row['winner']
            confidence = row['confidence_score']
            
            # Determine actual result (1 = gov wins, 0 = opp wins)
            if winner == "Government":
                actual_result = 1.0
            elif winner == "Opposition": 
                actual_result = 0.0
            else:
                continue  # Skip ties/errors
            
            # Weight by judge confidence (or use constant K-factor)
            if use_confidence:
                effective_k = self.k_factor * confidence
            else:
                effective_k = self.k_factor
            
            # Calculate expected results using ELO formula
            gov_elo = self.persona_elos[gov_persona]
            opp_elo = self.persona_elos[opp_persona]
            
            expected_gov = 1 / (1 + 10**((opp_elo - gov_elo) / 400))
            expected_opp = 1 - expected_gov
            
            # Update ELOs
            self.persona_elos[gov_persona] += effective_k * (actual_result - expected_gov)
            self.persona_elos[opp_persona] += effective_k * ((1 - actual_result) - expected_opp)
        
        return self.persona_elos
    
    def calculate_bear_bull_spectrum(self, df: pd.DataFrame, debate_topics: Dict[str, str] = None) -> Dict[str, float]:
        """Calculate bear-bull spectrum scores with even distribution"""
        
        # Get all personas sorted by name for consistency
        personas = sorted(self.persona_elos.keys())
        num_personas = len(personas)
        
        # Evenly distribute across bear-bull spectrum (-75 to +75)
        if num_personas == 1:
            scores = [0.0]
        else:
            # Create evenly spaced scores from -75 to +75
            scores = np.linspace(-75, 75, num_personas)
        
        # Assign scores to personas (alphabetically sorted for consistency)
        self.persona_bear_bull_scores = dict(zip(personas, scores))
        
        print(f"Bear-Bull spectrum assigned evenly across {num_personas} personas:")
        for persona, score in sorted(self.persona_bear_bull_scores.items(), key=lambda x: x[1]):
            position = "🐻 BEAR" if score < -25 else "🐂 BULL" if score > 25 else "⚖️ NEUTRAL"
            print(f"  {position} {persona}: {score:.1f}")
        
        return self.persona_bear_bull_scores
    
    def _determine_bull_bear_sides(self, gov_persona: str, opp_persona: str, topic: str) -> Tuple[str, str]:
        """Determine which persona is taking bull vs bear position based on names and topic"""
        
        # Bull indicators (optimistic/pro-growth personas)
        bull_indicators = ['bull', 'growth_bull', 'optimist', 'progressive', 'advocate']
        
        # Bear indicators (skeptical/conservative personas) 
        bear_indicators = ['bear', 'skeptic', 'conservative', 'realist', 'neutral']
        
        gov_is_bull = any(indicator in gov_persona.lower() for indicator in bull_indicators)
        gov_is_bear = any(indicator in gov_persona.lower() for indicator in bear_indicators)
        
        opp_is_bull = any(indicator in opp_persona.lower() for indicator in bull_indicators)
        opp_is_bear = any(indicator in opp_persona.lower() for indicator in bear_indicators)
        
        # Return bull_side, bear_side
        if gov_is_bull and opp_is_bear:
            return gov_persona, opp_persona
        elif gov_is_bear and opp_is_bull:
            return opp_persona, gov_persona
        elif gov_is_bull and not opp_is_bull:
            return gov_persona, opp_persona
        elif opp_is_bull and not gov_is_bull:
            return opp_persona, gov_persona
        elif gov_is_bear and not opp_is_bear:
            return opp_persona, gov_persona
        elif opp_is_bear and not gov_is_bear:
            return gov_persona, opp_persona
        else:
            # Fallback: use alphabetical ordering or return None
            return None, None
    
    def create_bear_bull_visualization(self, output_path: str = "bear_bull_spectrum.png"):
        """Create bear-bull spectrum visualization"""
        
        if not self.persona_bear_bull_scores or not self.persona_elos:
            raise ValueError("Must calculate ELO and bear-bull scores first")
        
        # Prepare data
        personas = list(self.persona_elos.keys())
        elos = [self.persona_elos[p] for p in personas]
        bear_bull_scores = [self.persona_bear_bull_scores[p] for p in personas]
        
        # Create the visualization
        plt.figure(figsize=(14, 8))
        
        # Create scatter plot
        scatter = plt.scatter(bear_bull_scores, elos, 
                            s=200, alpha=0.7, 
                            c=bear_bull_scores, cmap='RdYlGn',
                            edgecolors='black', linewidth=1)
        
        # Add persona labels
        for i, persona in enumerate(personas):
            plt.annotate(persona.replace('_', '\n'), 
                        (bear_bull_scores[i], elos[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Formatting
        plt.xlabel('Bear ← → Bull Spectrum', fontsize=14, fontweight='bold')
        plt.ylabel('ELO Rating (Debate Performance)', fontsize=14, fontweight='bold')
        plt.title('Persona Performance: Bear-Bull Spectrum vs Debate Skill', fontsize=16, fontweight='bold')
        
        # Add grid and reference lines
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Neutral Position')
        plt.axhline(y=1500, color='gray', linestyle='--', alpha=0.5, label='Starting ELO')
        
        # Color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Bear-Bull Score', rotation=270, labelpad=20)
        
        # Set axis limits with padding
        x_range = max(bear_bull_scores) - min(bear_bull_scores)
        y_range = max(elos) - min(elos)
        plt.xlim(min(bear_bull_scores) - x_range*0.1, max(bear_bull_scores) + x_range*0.1)
        plt.ylim(min(elos) - y_range*0.1, max(elos) + y_range*0.1)
        
        # Add legend
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Bear-Bull spectrum visualization saved to {output_path}")
    
    def generate_statistics_report(self) -> str:
        """Generate comprehensive statistics report"""
        
        if not self.persona_elos or not self.persona_bear_bull_scores:
            return "No data available. Run analysis first."
        
        # Sort personas by ELO
        sorted_by_elo = sorted(self.persona_elos.items(), key=lambda x: x[1], reverse=True)
        
        # Sort by bear-bull score
        sorted_by_spectrum = sorted(self.persona_bear_bull_scores.items(), key=lambda x: x[1])
        
        report = f"""
📊 DEBATE PERSONA ANALYSIS REPORT
{'='*50}

🏆 ELO RANKINGS (Debate Performance):
"""
        for i, (persona, elo) in enumerate(sorted_by_elo, 1):
            report += f"{i:2d}. {persona:20s} - {elo:7.1f} ELO\n"
        
        report += f"""
🐻→🐂 BEAR-BULL SPECTRUM (Evenly Distributed):
"""
        for persona, score in sorted_by_spectrum:
            position = "🐻 BEAR" if score < -25 else "🐂 BULL" if score > 25 else "⚖️ NEUTRAL"
            report += f"{position:12s} {persona:20s} - {score:6.1f} points\n"
        
        # Calculate correlations and insights
        elos_list = [self.persona_elos[p] for p in self.persona_elos.keys()]
        spectrum_list = [self.persona_bear_bull_scores[p] for p in self.persona_elos.keys()]
        
        correlation = np.corrcoef(elos_list, spectrum_list)[0, 1]
        
        report += f"""
📈 STATISTICAL INSIGHTS:
- Correlation between ELO and Bear-Bull position: {correlation:.3f}
- Average ELO: {np.mean(elos_list):.1f}
- ELO Standard Deviation: {np.std(elos_list):.1f}
- Average Bear-Bull Score: {np.mean(spectrum_list):.1f}
- Most Bearish: {min(self.persona_bear_bull_scores, key=self.persona_bear_bull_scores.get)} ({min(spectrum_list):.1f})
- Most Bullish: {max(self.persona_bear_bull_scores, key=self.persona_bear_bull_scores.get)} ({max(spectrum_list):.1f})
"""
        
        return report


def analyze_debate_results(results_dir: str, use_confidence: bool = True):
    """Main analysis function"""
    
    analyzer = DebateELOAnalyzer()
    
    # Load results
    df = analyzer.load_results(results_dir)
    
    # Calculate ELO ratings
    print("\n🏆 Calculating ELO ratings...")
    elos = analyzer.calculate_elo_ratings(df, use_confidence=use_confidence)
    
    # Calculate bear-bull spectrum
    print("\n🐻🐂 Calculating bear-bull spectrum...")
    debate_topics = {topic: topic for topic in df['topic'].unique()}
    spectrum = analyzer.calculate_bear_bull_spectrum(df, debate_topics)
    
    # Generate report
    print("\n📊 Generating analysis report...")
    report = analyzer.generate_statistics_report()
    print(report)
    
    # Create visualization
    print("\n📈 Creating bear-bull spectrum visualization...")
    confidence_suffix = "_with_confidence" if use_confidence else "_no_confidence"
    output_path = f"{results_dir}/bear_bull_spectrum{confidence_suffix}.png"
    analyzer.create_bear_bull_visualization(output_path)
    
    # Save detailed results
    results = {
        'elo_ratings': elos,
        'bear_bull_scores': spectrum,
        'analysis_report': report,
        'used_confidence_weighting': use_confidence
    }
    
    analysis_filename = f"elo_analysis{'_with_confidence' if use_confidence else '_no_confidence'}.json"
    with open(f"{results_dir}/{analysis_filename}", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to {results_dir}/")
    
    return analyzer


if __name__ == "__main__":
    # Analyze the results both ways
    results_dir = "balanced_debates_20251013_213432"
    
    print("🎯 ANALYSIS WITH CONFIDENCE WEIGHTING:")
    print("="*50)
    analyzer_with_confidence = analyze_debate_results(results_dir, use_confidence=True)
    
    print("\n" + "="*80)
    print("🎯 ANALYSIS WITHOUT CONFIDENCE WEIGHTING:")
    print("="*50)
    analyzer_no_confidence = analyze_debate_results(results_dir, use_confidence=False)
    
    print("\n" + "="*80)
    print("🔍 COMPARISON:")
    print("="*50)
    
    # Compare the two approaches
    with_conf = analyzer_with_confidence.persona_elos
    without_conf = analyzer_no_confidence.persona_elos
    
    print("ELO Differences (No Confidence - With Confidence):")
    for persona in sorted(with_conf.keys()):
        diff = without_conf[persona] - with_conf[persona]
        direction = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
        print(f"{direction} {persona:20s}: {diff:+6.1f}")
    
    print(f"\nAverage absolute difference: {np.mean([abs(without_conf[p] - with_conf[p]) for p in with_conf.keys()]):.1f} ELO points")