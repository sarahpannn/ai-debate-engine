# Our Retrieval Augmented Generation (RAG) System

<img src="diagram.png" alt="RAG System Diagram" width="700px"/>

Our pipeline augments debate-style LLM agents with grounded evidence from a vector RAG database. We embed documents (via the Gemini Embeddings API) and store them in a vector database. We supply our debater LLMs with the tools necessary to query the database and retrieve documents that are relevant to aiding their argumentation.. A separate Judge LLM scores the arguments on a rubric that we define; agents are then Elo-ranked to surface the strongest standpoints in debates around the stock.

# Content in this RAG Database

As specified in the compliance requirements for the Point72 pitch competition, all sources we use to augment the debate agents are publicly accessible and fall into one of the following categories.

* SEC filings  
  * 10-Ks and 10-Qs  
  * Retrieved with [https://github.com/sec-edgar/sec-edgar](https://github.com/sec-edgar/sec-edgar), processed with [https://github.com/alphanome-ai/sec-parser](https://github.com/alphanome-ai/sec-parser)	  
* Earnings call transcripts  
  * Used [fitz](https://pymupdf.readthedocs.io/en/latest/) for PDF OCR to extract text  
* Conference transcripts  
  * 2025 Goldman Sachs Communacopia \+ Technology Conference, 2024 RBC Capital Markets Global Technology, Internet, Media and Telecommunications Conference, 2025 Morgan Stanley Technology, Media, and Telecom Conference, 2025 27th Annual Needham Growth Conference, 2025 Cantor Fitzgerald Global Technology Conference, 2025 Piper Sandler 4th Annual Growth Frontiers Conference  
  * Used [fitz](https://pymupdf.readthedocs.io/en/latest/) for PDF OCR to extract text

# Quick Start

## Prerequisites

1. **Environment Variables**: Configure required API keys
   ```bash
   export ANTHROPIC_API_KEY="your_anthropic_key"
   export SUPABASE_URL="your_supabase_url"
   export SUPABASE_ANON_KEY="your_supabase_key" 
   export GEMINI_API_KEY="your_gemini_key"
   ```

2. **Install Dependencies**:
   ```bash
   pip install claudette anthropic supabase tiktoken loguru pandas matplotlib seaborn
   ```

## Running the System

### Single Test Debate
```bash
python3 test_toolloop.py
```
Executes one complete Parliamentary debate to verify system functionality.

### Continuous Debate Execution  
```bash
python3 continuous_debate_runner.py
```

The continuous runner:
- Generates all possible persona combinations across debate topics
- Randomly samples debates to ensure fair Elo rating distribution
- Utilizes RAG to search financial documents during speech generation
- Judges debates and updates Elo ratings using dynamic K-factor algorithm
- Saves progress and statistics to `debate_results/` directory
- Continues until API credit exhaustion or manual termination

### Results Analysis
```bash
python3 visualize_elo_ratings.py
```

Generates comprehensive visualizations including:
- Performance matrix heatmaps showing persona ratings across debate types
- Win rate analysis and statistical distributions
- Rating progression and experience correlation analysis
- Individual performance radar charts for top-performing personas

## System Architecture

**RAG Integration**: Debate personas access financial document database via `search_financial_documents` tool to retrieve supporting evidence for arguments.

**Parliamentary Debate Format**: Six-speech structure (Prime Minister, Leader of Opposition, Member of Government, Member of Opposition, Government Whip, Opposition Whip) with approximately 500 words per speech.

**Elo Rating System**: Implements dynamic K-factor adjustment based on persona experience level, with separate rating tracking per debate type.

**Supported Debate Topics**: Current implementation includes `ai_existential`, `climate_policy`, and `duo` debate configurations with specialized personas such as `pricing_power_bull`, `ai_risk_bear`, and `integration_pragmatist`.

For access to the vector database we used, email [sarahpan@mit.edu](mailto:sarahpan@mit.edu) for Supabase keys.

