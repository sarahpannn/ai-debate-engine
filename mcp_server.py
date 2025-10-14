"""
MCP Server for Debate Simulation
Provides tools for running debates, analyzing results, and managing personas.
"""

import asyncio
import json
import os
from typing import Any, Sequence

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    JSONSchema
)

from debate_engine import DebateEngine, DebateConfig, DebateMetrics
from batch_runner import BatchDebateRunner, BatchConfig, TopicLibrary
from llm_personas import PersonaLibrary
from claude_client import ClaudeDebateClient


class DebateSimulationMCP:
    """MCP Server for Debate Simulation Framework"""
    
    def __init__(self):
        self.server = Server("debate-simulation")
        self.debate_engine: DebateEngine = None
        self.batch_runner: BatchDebateRunner = None
        self.persona_library = PersonaLibrary()
        
        # Register tools
        self._register_tools()
        
    def _register_tools(self):
        """Register all MCP tools"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            return [
                Tool(
                    name="run_single_debate",
                    description="Run a single Parliamentary debate between two personas",
                    inputSchema=JSONSchema(
                        type="object",
                        properties={
                            "topic": {"type": "string", "description": "The debate topic"},
                            "government_persona": {"type": "string", "description": "Persona for government side"},
                            "opposition_persona": {"type": "string", "description": "Persona for opposition side"},
                            "max_words": {"type": "integer", "default": 500, "description": "Max words per speech"}
                        },
                        required=["topic", "government_persona", "opposition_persona"]
                    )
                ),
                Tool(
                    name="run_batch_debates",
                    description="Run a batch of debates for statistical analysis",
                    inputSchema=JSONSchema(
                        type="object",
                        properties={
                            "topics": {"type": "array", "items": {"type": "string"}, "description": "List of debate topics"},
                            "num_runs_per_topic": {"type": "integer", "default": 10, "description": "Runs per topic"},
                            "max_concurrent": {"type": "integer", "default": 3, "description": "Max concurrent debates"},
                            "output_dir": {"type": "string", "default": "debate_results", "description": "Output directory"}
                        },
                        required=["topics"]
                    )
                ),
                Tool(
                    name="list_personas",
                    description="List all available debate personas",
                    inputSchema=JSONSchema(type="object", properties={})
                ),
                Tool(
                    name="get_persona_details",
                    description="Get detailed information about a specific persona",
                    inputSchema=JSONSchema(
                        type="object",
                        properties={
                            "persona_name": {"type": "string", "description": "Name of the persona"}
                        },
                        required=["persona_name"]
                    )
                ),
                Tool(
                    name="analyze_results",
                    description="Analyze batch debate results and generate performance metrics",
                    inputSchema=JSONSchema(
                        type="object",
                        properties={
                            "results_dir": {"type": "string", "description": "Directory containing results"},
                            "timestamp": {"type": "string", "description": "Specific batch timestamp to analyze"}
                        },
                        required=["results_dir"]
                    )
                ),
                Tool(
                    name="get_default_topics",
                    description="Get default debate topics by category",
                    inputSchema=JSONSchema(
                        type="object",
                        properties={
                            "category": {
                                "type": "string", 
                                "enum": ["default", "political", "technology", "ethics"],
                                "default": "default",
                                "description": "Topic category"
                            }
                        }
                    )
                ),
                Tool(
                    name="setup_claude_client",
                    description="Initialize the Claude API client for debates",
                    inputSchema=JSONSchema(
                        type="object",
                        properties={
                            "api_key": {"type": "string", "description": "Claude API key"},
                            "model": {"type": "string", "default": "claude-3-5-sonnet-20241022", "description": "Claude model to use"}
                        },
                        required=["api_key"]
                    )
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            try:
                if name == "setup_claude_client":
                    return await self._setup_claude_client(arguments)
                elif name == "run_single_debate":
                    return await self._run_single_debate(arguments)
                elif name == "run_batch_debates":
                    return await self._run_batch_debates(arguments)
                elif name == "list_personas":
                    return await self._list_personas(arguments)
                elif name == "get_persona_details":
                    return await self._get_persona_details(arguments)
                elif name == "analyze_results":
                    return await self._analyze_results(arguments)
                elif name == "get_default_topics":
                    return await self._get_default_topics(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]
    
    async def _setup_claude_client(self, args: dict) -> list[TextContent]:
        """Setup Claude API client"""
        try:
            api_key = args["api_key"]
            model = args.get("model", "claude-3-5-sonnet-20241022")
            
            claude_client = ClaudeDebateClient(api_key=api_key, model=model)
            self.debate_engine = DebateEngine(claude_client, self.persona_library)
            self.batch_runner = BatchDebateRunner(self.debate_engine)
            
            # Validate setup
            is_valid = await self.debate_engine.validate_setup()
            
            if is_valid:
                return [TextContent(
                    type="text", 
                    text=f"✅ Claude client setup successful with model {model}\n"
                         f"Available personas: {len(self.persona_library.list_personas())}"
                )]
            else:
                return [TextContent(type="text", text="❌ Claude client validation failed")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Setup failed: {str(e)}")]
    
    async def _run_single_debate(self, args: dict) -> list[TextContent]:
        """Run a single debate"""
        if not self.debate_engine:
            return [TextContent(type="text", text="❌ Please setup Claude client first")]
        
        try:
            config = DebateConfig(
                topic=args["topic"],
                government_persona=args["government_persona"],
                opposition_persona=args["opposition_persona"],
                max_words_per_speech=args.get("max_words", 500)
            )
            
            result = await self.debate_engine.run_debate(config)
            
            if result.error_occurred:
                return [TextContent(type="text", text=f"❌ Debate failed: {result.error_message}")]
            
            # Format result
            output = f"""🏛️ DEBATE COMPLETED
            
**Topic:** {result.debate.topic}
**Government:** {result.debate.government_side} 
**Opposition:** {result.debate.opposition_side}
**Winner:** {result.winner}
**Judge Confidence:** {result.confidence_score:.3f}
**Duration:** {result.total_duration:.1f}s

**Judge Reasoning:**
{result.debate.judge_reasoning}

**Speech Summary:**
"""
            
            for i, speech in enumerate(result.debate.speeches, 1):
                side = "Gov" if any(speech.speaker.name == s.name for s in [result.debate.government_side]) else "Opp"
                output += f"{i}. {speech.speaker.value} ({speech.word_count} words)\n"
            
            return [TextContent(type="text", text=output)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Debate execution failed: {str(e)}")]
    
    async def _run_batch_debates(self, args: dict) -> list[TextContent]:
        """Run batch of debates"""
        if not self.batch_runner:
            return [TextContent(type="text", text="❌ Please setup Claude client first")]
        
        try:
            config = BatchConfig(
                topics=args["topics"],
                num_runs_per_topic=args.get("num_runs_per_topic", 10),
                max_concurrent_debates=args.get("max_concurrent", 3),
                output_dir=args.get("output_dir", "debate_results")
            )
            
            batch_results = await self.batch_runner.run_batch(config)
            
            # Format summary
            report = batch_results.performance_report
            output = f"""📊 BATCH RESULTS SUMMARY

**Execution Time:** {batch_results.execution_time:.1f}s
**Total Debates:** {report['total_debates']}
**Successful:** {report['successful_debates']}
**Error Rate:** {report['error_rate']:.1%}

**Performance Metrics:**
- Average Duration: {report['average_duration_seconds']:.1f}s
- Average Judge Confidence: {report['average_judge_confidence']:.3f}

**Side Bias:**
- Government Win Rate: {report['side_bias']['government']:.1%}
- Opposition Win Rate: {report['side_bias']['opposition']:.1%}

**Top Performers:**
"""
            
            for persona, win_rate in sorted(report['persona_win_rates'].items(), key=lambda x: x[1], reverse=True)[:5]:
                output += f"- {persona}: {win_rate:.1%}\n"
            
            output += f"\n**Results saved to:** {config.output_dir}/"
            
            return [TextContent(type="text", text=output)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Batch execution failed: {str(e)}")]
    
    async def _list_personas(self, args: dict) -> list[TextContent]:
        """List available personas"""
        personas = self.persona_library.list_personas()
        
        output = "🎭 AVAILABLE DEBATE PERSONAS\n\n"
        for persona_key in personas:
            persona = self.persona_library.get_persona(persona_key)
            output += f"**{persona.name}** \n"
            output += f"- Characteristics: {', '.join(persona.characteristics)}\n"
            output += f"- Temperature: {persona.temperature}\n\n"
        
        return [TextContent(type="text", text=output)]
    
    async def _get_persona_details(self, args: dict) -> list[TextContent]:
        """Get detailed persona information"""
        persona_name = args["persona_name"]
        persona = self.persona_library.get_persona(persona_name)
        
        if not persona:
            return [TextContent(type="text", text=f"❌ Persona '{persona_name}' not found")]
        
        output = f"""🎭 PERSONA DETAILS: {persona.name}

**Temperature:** {persona.temperature}
**Max Tokens:** {persona.max_tokens}

**Characteristics:**
{chr(10).join(f"- {char}" for char in persona.characteristics)}

**System Prompt:**
{persona.system_prompt}
"""
        
        return [TextContent(type="text", text=output)]
    
    async def _analyze_results(self, args: dict) -> list[TextContent]:
        """Analyze batch results"""
        try:
            results_dir = args["results_dir"]
            timestamp = args.get("timestamp")
            
            if timestamp:
                summary_file = f"{results_dir}/batch_summary_{timestamp}.json"
                csv_file = f"{results_dir}/batch_results_{timestamp}.csv"
            else:
                # Find most recent results
                import glob
                summary_files = glob.glob(f"{results_dir}/batch_summary_*.json")
                if not summary_files:
                    return [TextContent(type="text", text=f"❌ No results found in {results_dir}")]
                summary_file = max(summary_files)
                timestamp = summary_file.split("_")[-1].replace(".json", "")
                csv_file = f"{results_dir}/batch_results_{timestamp}.csv"
            
            # Load and analyze
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            import pandas as pd
            df = pd.read_csv(csv_file)
            
            output = f"""📈 DETAILED ANALYSIS - {timestamp}

**Dataset Overview:**
- Total Debates: {len(df)}
- Success Rate: {(1 - df['error_occurred'].mean()):.1%}
- Topics: {len(df['topic'].unique())}
- Unique Personas: {len(set(df['government_persona'].unique()) | set(df['opposition_persona'].unique()))}

**Performance Distribution:**
- Avg Confidence: {df['confidence_score'].mean():.3f} ± {df['confidence_score'].std():.3f}
- Avg Duration: {df['total_duration'].mean():.1f}s ± {df['total_duration'].std():.1f}s
- Avg Words/Debate: {df['total_words'].mean():.0f}

**Win Rate Rankings:**
"""
            
            # Calculate detailed win rates
            personas = set(df['government_persona'].unique()) | set(df['opposition_persona'].unique())
            win_rates = {}
            for persona in personas:
                gov_wins = ((df['government_persona'] == persona) & (df['winner'] == 'Government')).sum()
                opp_wins = ((df['opposition_persona'] == persona) & (df['winner'] == 'Opposition')).sum()
                total_games = ((df['government_persona'] == persona) | (df['opposition_persona'] == persona)).sum()
                if total_games > 0:
                    win_rates[persona] = (gov_wins + opp_wins) / total_games
            
            for persona, rate in sorted(win_rates.items(), key=lambda x: x[1], reverse=True):
                output += f"- {persona}: {rate:.1%}\n"
            
            return [TextContent(type="text", text=output)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Analysis failed: {str(e)}")]
    
    async def _get_default_topics(self, args: dict) -> list[TextContent]:
        """Get default topics by category"""
        category = args.get("category", "default")
        
        topic_map = {
            "default": TopicLibrary.get_default_topics(),
            "political": TopicLibrary.get_political_topics(),
            "technology": TopicLibrary.get_technology_topics(),
            "ethics": TopicLibrary.get_ethics_topics()
        }
        
        topics = topic_map.get(category, TopicLibrary.get_default_topics())
        
        output = f"📋 {category.upper()} DEBATE TOPICS\n\n"
        for i, topic in enumerate(topics, 1):
            output += f"{i}. {topic}\n"
        
        return [TextContent(type="text", text=output)]

    async def run(self):
        """Run the MCP server"""
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="debate-simulation",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main():
    """Main entry point"""
    mcp_server = DebateSimulationMCP()
    await mcp_server.run()


if __name__ == "__main__":
    asyncio.run(main())