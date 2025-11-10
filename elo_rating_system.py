"""
Elo Rating System for Debate Personas
Tracks performance of different personas across debates.
"""

import json
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EloRating:
    persona_id: str
    debate_name: str
    rating: float
    games_played: int
    wins: int
    losses: int
    
    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    def to_dict(self) -> Dict:
        return {
            "persona_id": self.persona_id,
            "debate_name": self.debate_name,
            "rating": round(self.rating, 2),
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 3)
        }


class EloRatingSystem:
    """
    Elo rating system for tracking debate persona performance.
    
    Standard Elo with K-factor adjustment based on games played.
    """
    
    def __init__(self, initial_rating: float = 1500, k_factor: float = 32):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: Dict[str, EloRating] = {}  # key: f"{debate_name}:{persona_id}"
        
    def _get_rating_key(self, debate_name: str, persona_id: str) -> str:
        """Generate key for rating lookup"""
        return f"{debate_name}:{persona_id}"
    
    def get_rating(self, debate_name: str, persona_id: str) -> EloRating:
        """Get current rating for a persona in a specific debate type"""
        key = self._get_rating_key(debate_name, persona_id)
        
        if key not in self.ratings:
            self.ratings[key] = EloRating(
                persona_id=persona_id,
                debate_name=debate_name,
                rating=self.initial_rating,
                games_played=0,
                wins=0,
                losses=0
            )
        
        return self.ratings[key]
    
    def calculate_expected_score(self, rating_a: float, rating_b: float) -> Tuple[float, float]:
        """Calculate expected scores for two players"""
        expected_a = 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
        expected_b = 1 / (1 + math.pow(10, (rating_a - rating_b) / 400))
        return expected_a, expected_b
    
    def get_k_factor(self, games_played: int) -> float:
        """Dynamic K-factor based on games played"""
        if games_played < 10:
            return self.k_factor * 1.5  # Higher for new players
        elif games_played < 30:
            return self.k_factor
        else:
            return self.k_factor * 0.75  # Lower for experienced players
    
    def update_ratings(self, debate_name: str, gov_persona: str, opp_persona: str, 
                      winner: str) -> Tuple[EloRating, EloRating]:
        """
        Update ratings based on debate result.
        
        Args:
            debate_name: The debate configuration name
            gov_persona: Government persona ID
            opp_persona: Opposition persona ID
            winner: "Government" or "Opposition"
            
        Returns:
            Tuple of updated ratings (government, opposition)
        """
        
        # Get current ratings
        gov_rating = self.get_rating(debate_name, gov_persona)
        opp_rating = self.get_rating(debate_name, opp_persona)
        
        # Calculate expected scores
        expected_gov, expected_opp = self.calculate_expected_score(
            gov_rating.rating, opp_rating.rating
        )
        
        # Determine actual scores (1 for win, 0 for loss)
        if winner == "Government":
            actual_gov, actual_opp = 1.0, 0.0
            gov_rating.wins += 1
            opp_rating.losses += 1
        elif winner == "Opposition":
            actual_gov, actual_opp = 0.0, 1.0
            gov_rating.losses += 1
            opp_rating.wins += 1
        else:
            # Handle ties or errors - no rating change
            gov_rating.games_played += 1
            opp_rating.games_played += 1
            return gov_rating, opp_rating
        
        # Get dynamic K-factors
        k_gov = self.get_k_factor(gov_rating.games_played)
        k_opp = self.get_k_factor(opp_rating.games_played)
        
        # Update ratings
        gov_rating.rating += k_gov * (actual_gov - expected_gov)
        opp_rating.rating += k_opp * (actual_opp - expected_opp)
        
        # Update games played
        gov_rating.games_played += 1
        opp_rating.games_played += 1
        
        return gov_rating, opp_rating
    
    def get_leaderboard(self, debate_name: Optional[str] = None, 
                       min_games: int = 5) -> List[EloRating]:
        """
        Get leaderboard sorted by rating.
        
        Args:
            debate_name: Filter by specific debate type, or None for all
            min_games: Minimum games played to be included
            
        Returns:
            List of EloRating objects sorted by rating (descending)
        """
        filtered_ratings = []
        
        for rating in self.ratings.values():
            if rating.games_played < min_games:
                continue
            
            if debate_name is not None and rating.debate_name != debate_name:
                continue
                
            filtered_ratings.append(rating)
        
        # Sort by rating (descending), then by games played (descending)
        return sorted(filtered_ratings, key=lambda r: (r.rating, r.games_played), reverse=True)
    
    def get_persona_summary(self, persona_id: str) -> Dict[str, EloRating]:
        """Get all ratings for a specific persona across debate types"""
        persona_ratings = {}
        
        for rating in self.ratings.values():
            if rating.persona_id == persona_id:
                persona_ratings[rating.debate_name] = rating
        
        return persona_ratings
    
    def save_ratings(self, file_path: Path):
        """Save ratings to JSON file"""
        ratings_data = {
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "ratings": [rating.to_dict() for rating in self.ratings.values()]
        }
        
        with open(file_path, 'w') as f:
            json.dump(ratings_data, f, indent=2)
    
    def load_ratings(self, file_path: Path):
        """Load ratings from JSON file"""
        if not file_path.exists():
            return
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.initial_rating = data.get("initial_rating", 1500)
        self.k_factor = data.get("k_factor", 32)
        
        self.ratings = {}
        for rating_data in data.get("ratings", []):
            rating = EloRating(
                persona_id=rating_data["persona_id"],
                debate_name=rating_data["debate_name"],
                rating=rating_data["rating"],
                games_played=rating_data["games_played"],
                wins=rating_data["wins"],
                losses=rating_data["losses"]
            )
            key = self._get_rating_key(rating.debate_name, rating.persona_id)
            self.ratings[key] = rating
    
    def print_leaderboard(self, debate_name: Optional[str] = None, limit: int = 20):
        """Print formatted leaderboard"""
        leaderboard = self.get_leaderboard(debate_name, min_games=3)[:limit]
        
        if not leaderboard:
            print("No ratings available (minimum 3 games required)")
            return
        
        title = f"ELO LEADERBOARD - {debate_name or 'ALL DEBATES'}"
        print(f"\n{title}")
        print("=" * len(title))
        print(f"{'Rank':<4} {'Persona':<20} {'Debate':<15} {'Rating':<8} {'W-L':<8} {'Games':<6} {'Win%':<6}")
        print("-" * 70)
        
        for i, rating in enumerate(leaderboard, 1):
            print(f"{i:<4} {rating.persona_id:<20} {rating.debate_name:<15} "
                  f"{rating.rating:<8.0f} {rating.wins}-{rating.losses:<7} "
                  f"{rating.games_played:<6} {rating.win_rate*100:<6.1f}%")
    
    def get_rating_distribution(self, debate_name: Optional[str] = None) -> Dict[str, int]:
        """Get distribution of ratings by tier"""
        ratings = self.get_leaderboard(debate_name, min_games=1)
        
        distribution = {
            "Grandmaster (2400+)": 0,
            "Master (2200-2399)": 0,
            "Expert (2000-2199)": 0,
            "Advanced (1800-1999)": 0,
            "Intermediate (1600-1799)": 0,
            "Novice (1400-1599)": 0,
            "Beginner (<1400)": 0
        }
        
        for rating in ratings:
            r = rating.rating
            if r >= 2400:
                distribution["Grandmaster (2400+)"] += 1
            elif r >= 2200:
                distribution["Master (2200-2399)"] += 1
            elif r >= 2000:
                distribution["Expert (2000-2199)"] += 1
            elif r >= 1800:
                distribution["Advanced (1800-1999)"] += 1
            elif r >= 1600:
                distribution["Intermediate (1600-1799)"] += 1
            elif r >= 1400:
                distribution["Novice (1400-1599)"] += 1
            else:
                distribution["Beginner (<1400)"] += 1
        
        return distribution