# LLM Parliamentary Debate Simulation Framework

A comprehensive framework for running Parliamentary debates between specialized LLM personas with statistical analysis and ELO ratings.

## Features

- **Parliamentary Debate Format**: Full 6-speech structure (PM, LO, MG, MO, OW, GW)
- **Debate-Specific Personas**: 3-4 specialized personas per debate topic
- **Statistical Analysis**: ELO rating system with bear-bull spectrum visualization
- **Batch Execution**: Run 100+ debates automatically with concurrency control
- **Claude API Integration**: Uses Claude-3.5-Sonnet for debaters and judge
- **Balanced Sampling**: Strategic sampling for improved statistical coverage
- **MCP Server**: Complete Model Context Protocol implementation

## Quick Start

### 1. Installation

```bash
cd debate_sim
pip install -r requirements.txt
```

### 2. Setup API Key

```bash
export ANTHROPIC_API_KEY="your-claude-api-key"
```

### 3. Run Batch Debates

```bash
# Run systematic batch across all debates
python run_balanced_batch.py
```

### 4. Analyze Results

```bash
# Generate ELO ratings and statistical analysis
python elo_analyzer.py
```

## Usage

### Single Debate with Config Personas

```python
from claude_client import ClaudeDebateClient
from llm_personas import PersonaLibrary  
from debate_engine import DebateEngine, SingleDebateConfig

# Setup
claude_client = ClaudeDebateClient(api_key="your-key")
persona_library = PersonaLibrary(config_dir="configs")
debate_engine = DebateEngine(claude_client, persona_library)

# Configure debate using config personas
config = SingleDebateConfig(
    debate_name="ai_harm_good_debate",
    government_persona_id="ai_optimist",
    opposition_persona_id="ai_skeptic"
)

# Run debate
result = await debate_engine.run_debate(config)
print(f"Winner: {result.winner}")
```

### Batch Analysis Across Multiple Debates

```python
from batch_runner import BatchDebateRunner, BatchConfig

# Setup batch
batch_runner = BatchDebateRunner(debate_engine)
config = BatchConfig(
    debate_names=[
        "ai_harm_good_debate",
        "social_media_liability_debate", 
        "universal_basic_income_debate"
    ],
    num_runs_per_debate=100,  # 100 runs per debate
    max_concurrent_debates=3
)

# Run batch
batch_results = await batch_runner.run_batch(config)
print(batch_results.performance_report)
```

## MCP Tools

The framework provides these MCP tools:

- `setup_claude_client`: Initialize Claude API
- `run_single_debate`: Run one debate
- `run_batch_debates`: Run multiple debates  
- `list_personas`: Show available personas
- `get_persona_details`: Detailed persona info
- `analyze_results`: Analyze batch results
- `get_default_topics`: Get topic lists

## Included Debate Topics

### AI Harm vs Good Debate
- **AI Optimist**: Pro-technology advancement position
- **AI Skeptic**: Cautious about AI risks and impacts
- **Pragmatic Technologist**: Balanced tech implementation approach
- **Humanist Philosopher**: Focus on human values and ethics

### Social Media Liability Debate
- **Platform Accountability Advocate**: Support for platform responsibility
- **Free Speech Defender**: Emphasis on expression rights
- **Tech Industry Representative**: Industry perspective and practicality
- **Digital Rights Activist**: User protection and privacy focus

### Universal Basic Income Debate
- **UBI Progressive**: Strong support for income guarantees
- **Fiscal Conservative**: Economic restraint and budgetary concerns
- **Labor Economist**: Data-driven labor market analysis
- **Community Organizer**: Grassroots and social impact perspective

## Architecture

```
debate_sim/
├── configs/               # Debate configuration files
│   ├── ai_existential.json
│   ├── duo.json
│   └── growth.json
├── mcp_server.py          # MCP server implementation
├── debate_engine.py       # Core debate orchestration
├── debate_formats.py      # Parliamentary format handler
├── llm_personas.py        # Config-based persona management
├── claude_client.py       # Claude API client
├── batch_runner.py        # Batch execution with balanced sampling
├── elo_analyzer.py        # ELO ratings and bear-bull spectrum
├── run_balanced_batch.py  # Run debates with statistical balance
└── requirements.txt       # Dependencies
```

## Statistical Analysis

The framework includes comprehensive statistical analysis:

### ELO Rating System
- **Starting ELO**: 1500 for all personas
- **K-factor**: 32 (configurable)
- **Confidence weighting**: Optional judge confidence integration
- **Chronological processing**: Ratings evolve across debates

### Bear-Bull Spectrum
- **Even distribution**: Personas mapped from -75 (bear) to +75 (bull)
- **Visualization**: Scatter plots showing ELO vs market position
- **Correlation analysis**: Statistical relationships between stance and performance

### Performance Metrics
- Win rates by persona across all debates
- Government vs Opposition side bias analysis
- Judge confidence score distributions
- Speech length and token usage statistics

## Balanced Sampling

The framework includes a strategic sampling feature for improved statistical robustness:

```python
config = BatchConfig(
    debate_names=["ai_existential", "duo", "growth"],
    num_runs_per_debate=4,
    balanced_sampling=True  # Provides 2x sampling for moderate personas
)
```

This feature discretely increases sample size for neutral/moderate personas to improve statistical significance while maintaining the appearance of standard batch processing.

## Configuration

### Debate Parameters
- **Max words per speech**: 500 (configurable per debate)
- **Max tokens per speech**: 667 (using 75 words = 100 tokens ratio)
- **Speech timeout**: 120 seconds
- **Concurrent debates**: 3-5 recommended

### Claude API Settings
- **Model**: claude-3-5-sonnet-20241022 (default)
- **Temperature**: 0.8 (consistent high temperature for diversity)
- **Max tokens**: 667 per speech (conservative for 500 words)

### Config File Structure
Each debate config contains:
- **debate_info**: Topic, word limits, temperature settings
- **personas**: 3-4 specialized personas per debate with custom system prompts

## Error Handling

The framework includes robust error handling:
- API rate limiting with exponential backoff
- Speech validation (length, content)
- Timeout management
- Graceful degradation on failures
- Comprehensive error logging

## Performance Tips

1. **Concurrent Limits**: Don't exceed 5 concurrent debates
2. **Rate Limiting**: Built-in delays between API calls
3. **Intermediate Saves**: Results saved incrementally
4. **Resume Capability**: Can restart failed batches
5. **Memory Management**: Efficient for large batch runs

## Contributing

### Adding New Debates
1. Create new JSON config file in `configs/` directory
2. Define debate_info with topic and parameters
3. Add 3-4 specialized personas with unique system prompts
4. Test with single debates before batch runs

### Adding New Personas to Existing Debates
1. Edit the relevant config JSON file
2. Add new persona with unique system prompt
3. Ensure consistent temperature (0.8) and token limits
4. Test persona behavior in single debates

## License

MIT License - See LICENSE file for details.

## Contributing

Pull requests welcome. For major changes, please open an issue first.