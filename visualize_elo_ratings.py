"""
Elo Rating Visualization Script
Creates interactive plots of persona performance across different debate types.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import seaborn as sns


def load_elo_data(file_path: str = "debate_results/elo_ratings.json") -> Dict:
    """Load Elo rating data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_elo_dataframe(elo_data: Dict) -> pd.DataFrame:
    """Convert Elo data to pandas DataFrame for easier analysis"""
    ratings = elo_data.get('ratings', [])
    return pd.DataFrame(ratings)


def create_radar_chart(df: pd.DataFrame, persona_id: str):
    """Create a radar chart showing persona performance across debates"""
    persona_data = df[df['persona_id'] == persona_id]
    
    if len(persona_data) == 0:
        print(f"No data found for persona: {persona_id}")
        return
    
    # Get debate types and ratings
    debates = persona_data['debate_name'].tolist()
    ratings = persona_data['rating'].tolist()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Calculate angles for each debate
    angles = np.linspace(0, 2 * np.pi, len(debates), endpoint=False).tolist()
    
    # Close the plot
    ratings += ratings[:1]
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, ratings, 'o-', linewidth=2, label=persona_id)
    ax.fill(angles, ratings, alpha=0.25)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(debates)
    ax.set_ylim(1200, 2000)  # Typical Elo range
    ax.set_title(f"{persona_id} - Elo Ratings Across Debates", size=16, y=1.1)
    
    # Add grid
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def create_debate_comparison_plot(df: pd.DataFrame):
    """Create a plot comparing all personas within each debate type"""
    debate_types = df['debate_name'].unique()
    
    fig, axes = plt.subplots(1, len(debate_types), figsize=(6*len(debate_types), 8))
    
    if len(debate_types) == 1:
        axes = [axes]
    
    for i, debate_type in enumerate(debate_types):
        debate_data = df[df['debate_name'] == debate_type]
        
        # Sort by rating
        debate_data = debate_data.sort_values('rating', ascending=True)
        
        # Create bar plot
        bars = axes[i].barh(range(len(debate_data)), debate_data['rating'])
        
        # Color bars by performance
        for j, (idx, row) in enumerate(debate_data.iterrows()):
            if row['rating'] > 1600:
                bars[j].set_color('green')
            elif row['rating'] > 1400:
                bars[j].set_color('orange')
            else:
                bars[j].set_color('red')
        
        # Customize axis
        axes[i].set_yticks(range(len(debate_data)))
        axes[i].set_yticklabels(debate_data['persona_id'])
        axes[i].set_xlabel('Elo Rating')
        axes[i].set_title(f'{debate_type}\n(Games Played)')
        axes[i].grid(axis='x', alpha=0.3)
        
        # Add games played annotations
        for j, (idx, row) in enumerate(debate_data.iterrows()):
            axes[i].text(row['rating'] + 10, j, f"{row['games_played']}g", 
                        va='center', fontsize=8)
    
    plt.suptitle('Elo Ratings by Debate Type', fontsize=16)
    plt.tight_layout()
    return fig


def create_performance_matrix(df: pd.DataFrame):
    """Create a heatmap matrix showing persona performance across debates"""
    # Pivot the data to get persona vs debate matrix
    matrix = df.pivot(index='persona_id', columns='debate_name', values='rating')
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    # Use a color map that makes differences clear
    sns.heatmap(matrix, 
                annot=True, 
                fmt='.0f',
                cmap='RdYlGn',
                center=1500,  # Center at default Elo
                cbar_kws={'label': 'Elo Rating'})
    
    plt.title('Persona Performance Matrix\n(Elo Ratings Across Debate Types)', fontsize=14)
    plt.xlabel('Debate Type')
    plt.ylabel('Persona')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return plt.gcf()


def create_win_rate_comparison(df: pd.DataFrame):
    """Create a comparison of win rates across personas and debates"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Win rate by persona (average across all debates)
    persona_stats = df.groupby('persona_id').agg({
        'win_rate': 'mean',
        'games_played': 'sum'
    }).sort_values('win_rate', ascending=True)
    
    bars1 = ax1.barh(range(len(persona_stats)), persona_stats['win_rate'])
    
    # Color by win rate
    for i, rate in enumerate(persona_stats['win_rate']):
        if rate > 0.6:
            bars1[i].set_color('darkgreen')
        elif rate > 0.4:
            bars1[i].set_color('orange')
        else:
            bars1[i].set_color('darkred')
    
    ax1.set_yticks(range(len(persona_stats)))
    ax1.set_yticklabels(persona_stats.index)
    ax1.set_xlabel('Average Win Rate')
    ax1.set_title('Average Win Rate by Persona')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add games played annotations
    for i, (persona, stats) in enumerate(persona_stats.iterrows()):
        ax1.text(stats['win_rate'] + 0.01, i, f"{stats['games_played']}g", 
                va='center', fontsize=8)
    
    # Plot 2: Win rate by debate type
    debate_stats = df.groupby('debate_name').agg({
        'win_rate': 'mean',
        'games_played': 'sum'
    }).sort_values('win_rate', ascending=True)
    
    bars2 = ax2.barh(range(len(debate_stats)), debate_stats['win_rate'])
    
    for i, rate in enumerate(debate_stats['win_rate']):
        if rate > 0.55:
            bars2[i].set_color('darkgreen')
        elif rate > 0.45:
            bars2[i].set_color('orange')
        else:
            bars2[i].set_color('darkred')
    
    ax2.set_yticks(range(len(debate_stats)))
    ax2.set_yticklabels(debate_stats.index)
    ax2.set_xlabel('Average Win Rate')
    ax2.set_title('Average Win Rate by Debate Type')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add total games annotations
    for i, (debate, stats) in enumerate(debate_stats.iterrows()):
        ax2.text(stats['win_rate'] + 0.01, i, f"{stats['games_played']}g", 
                va='center', fontsize=8)
    
    plt.suptitle('Win Rate Analysis', fontsize=16)
    plt.tight_layout()
    return fig


def create_elo_distribution_plot(df: pd.DataFrame):
    """Create distribution plots of Elo ratings"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Overall Elo distribution
    axes[0,0].hist(df['rating'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(df['rating'].mean(), color='red', linestyle='--', label=f'Mean: {df["rating"].mean():.0f}')
    axes[0,0].set_xlabel('Elo Rating')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Overall Elo Rating Distribution')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    
    # Plot 2: Elo by debate type (box plot)
    debate_ratings = [df[df['debate_name'] == debate]['rating'].values 
                     for debate in df['debate_name'].unique()]
    
    axes[0,1].boxplot(debate_ratings, labels=df['debate_name'].unique())
    axes[0,1].set_ylabel('Elo Rating')
    axes[0,1].set_title('Elo Distribution by Debate Type')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(alpha=0.3)
    
    # Plot 3: Games played vs Rating (scatter)
    scatter = axes[1,0].scatter(df['games_played'], df['rating'], 
                               c=df['win_rate'], cmap='RdYlGn', alpha=0.7)
    axes[1,0].set_xlabel('Games Played')
    axes[1,0].set_ylabel('Elo Rating')
    axes[1,0].set_title('Rating vs Experience\n(Color = Win Rate)')
    plt.colorbar(scatter, ax=axes[1,0], label='Win Rate')
    axes[1,0].grid(alpha=0.3)
    
    # Plot 4: Rating vs Win Rate (scatter)
    for debate in df['debate_name'].unique():
        debate_data = df[df['debate_name'] == debate]
        axes[1,1].scatter(debate_data['rating'], debate_data['win_rate'], 
                         label=debate, alpha=0.7)
    
    axes[1,1].set_xlabel('Elo Rating')
    axes[1,1].set_ylabel('Win Rate')
    axes[1,1].set_title('Rating vs Win Rate by Debate')
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_summary_stats(df: pd.DataFrame):
    """Generate and print summary statistics"""
    print("="*60)
    print("ELO RATING ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"Total personas: {df['persona_id'].nunique()}")
    print(f"Total debate types: {df['debate_name'].nunique()}")
    print(f"Total games played: {df['games_played'].sum()}")
    print(f"Average Elo rating: {df['rating'].mean():.1f}")
    print(f"Elo range: {df['rating'].min():.0f} - {df['rating'].max():.0f}")
    
    print(f"\n🏆 TOP PERFORMERS:")
    top_performers = df.nlargest(5, 'rating')[['persona_id', 'debate_name', 'rating', 'win_rate', 'games_played']]
    for _, row in top_performers.iterrows():
        print(f"  {row['persona_id']} ({row['debate_name']}): {row['rating']:.0f} Elo, {row['win_rate']:.1%} win rate, {row['games_played']} games")
    
    print(f"\n📉 STRUGGLING PERSONAS:")
    bottom_performers = df.nsmallest(5, 'rating')[['persona_id', 'debate_name', 'rating', 'win_rate', 'games_played']]
    for _, row in bottom_performers.iterrows():
        print(f"  {row['persona_id']} ({row['debate_name']}): {row['rating']:.0f} Elo, {row['win_rate']:.1%} win rate, {row['games_played']} games")
    
    print(f"\n🎯 BY DEBATE TYPE:")
    for debate in df['debate_name'].unique():
        debate_data = df[df['debate_name'] == debate]
        avg_rating = debate_data['rating'].mean()
        total_games = debate_data['games_played'].sum()
        print(f"  {debate}: {avg_rating:.0f} avg Elo, {total_games} total games")


def main():
    """Main function to generate all visualizations"""
    
    # Check if file exists
    elo_file = "debate_results/elo_ratings.json"
    if not Path(elo_file).exists():
        print(f"❌ Elo ratings file not found: {elo_file}")
        print("Run some debates first to generate rating data!")
        return
    
    # Load and process data
    print(f"📊 Loading Elo ratings from {elo_file}...")
    elo_data = load_elo_data(elo_file)
    df = create_elo_dataframe(elo_data)
    
    if len(df) == 0:
        print("❌ No rating data found in file!")
        return
    
    print(f"✅ Loaded {len(df)} rating records")
    
    # Generate summary stats
    generate_summary_stats(df)
    
    # Create output directory
    output_dir = Path("debate_results/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🎨 Generating visualizations...")
    
    # 1. Performance matrix heatmap
    print("  📈 Creating performance matrix...")
    fig1 = create_performance_matrix(df)
    fig1.savefig(output_dir / "elo_performance_matrix.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Debate comparison
    print("  🥊 Creating debate comparison...")
    fig2 = create_debate_comparison_plot(df)
    fig2.savefig(output_dir / "elo_debate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Win rate analysis
    print("  🏆 Creating win rate analysis...")
    fig3 = create_win_rate_comparison(df)
    fig3.savefig(output_dir / "elo_win_rates.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Distribution analysis
    print("  📊 Creating distribution analysis...")
    fig4 = create_elo_distribution_plot(df)
    fig4.savefig(output_dir / "elo_distributions.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Individual radar charts for top personas
    print("  🎯 Creating individual persona radar charts...")
    top_personas = df.groupby('persona_id')['rating'].mean().nlargest(3).index
    
    for persona in top_personas:
        if df[df['persona_id'] == persona]['debate_name'].nunique() > 1:  # Only if they have multiple debates
            fig = create_radar_chart(df, persona)
            if fig:
                fig.savefig(output_dir / f"radar_{persona.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    print(f"\n✅ All visualizations saved to {output_dir}/")
    print(f"📁 Generated files:")
    for file in output_dir.glob("*.png"):
        print(f"  - {file.name}")
    
    print(f"\n💡 Pro tip: Open the PNG files to see your Elo rating analysis!")


if __name__ == "__main__":
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    main()