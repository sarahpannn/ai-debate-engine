"""
Format debate JSON files to be more screenshot-friendly by converting \\n to actual line breaks
and reformatting the speech content for better readability.
"""

import json
import textwrap
from pathlib import Path


def format_speech_content(content: str, width: int = 80) -> str:
    """Format speech content with proper line breaks and wrapping"""
    # Replace \\n with actual line breaks
    content = content.replace('\\n', '\n')
    
    # Split into paragraphs
    paragraphs = content.split('\n\n')
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        # Remove single line breaks within paragraphs and rewrap
        paragraph = paragraph.replace('\n', ' ').strip()
        if paragraph:
            wrapped = textwrap.fill(paragraph, width=width)
            formatted_paragraphs.append(wrapped)
    
    return '\n\n'.join(formatted_paragraphs)


def format_debate_json(input_file: str, output_file: str = None, width: int = 80):
    """Format a debate JSON file for better readability"""
    
    if output_file is None:
        output_file = input_file.replace('.json', '_formatted.json')
    
    # Read the original JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Format speech content
    if 'speeches' in data:
        for speech in data['speeches']:
            if 'content' in speech:
                speech['content'] = format_speech_content(speech['content'], width)
    
    # Format judge reasoning if present
    if 'judge_reasoning' in data and data['judge_reasoning']:
        data['judge_reasoning'] = format_speech_content(data['judge_reasoning'], width)
    
    # Write formatted JSON with proper indentation
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Formatted {input_file} -> {output_file}")
    return output_file


def format_all_debates_in_directory(directory: str, width: int = 80):
    """Format all debate JSON files in a directory"""
    
    directory = Path(directory)
    debate_files = list(directory.glob("debate_*.json"))
    
    if not debate_files:
        print(f"❌ No debate JSON files found in {directory}")
        return
    
    print(f"🔄 Formatting {len(debate_files)} debate files...")
    
    formatted_files = []
    for file_path in sorted(debate_files):
        output_path = str(file_path).replace('.json', '_formatted.json')
        format_debate_json(str(file_path), output_path, width)
        formatted_files.append(output_path)
    
    print(f"\n📋 Summary:")
    print(f"   Original files: {len(debate_files)}")
    print(f"   Formatted files: {len(formatted_files)}")
    print(f"   Text width: {width} characters")
    print(f"   Output directory: {directory}")
    
    # Show example of first formatted file
    if formatted_files:
        print(f"\n📄 Example formatted file: {Path(formatted_files[0]).name}")
        
    return formatted_files


if __name__ == "__main__":
    # Format the specific file you mentioned
    input_file = "all_debates_results_20251013_173433/debate_20251013_173433_0000.json"
    
    print("🎯 Formatting debate JSON for screenshot-friendly display...")
    print("="*60)
    
    # Format single file with different widths
    format_debate_json(input_file, width=70)
    
    # Also format all files in the directory
    print(f"\n🔄 Formatting all debate files in directory...")
    format_all_debates_in_directory("all_debates_results_20251013_173433", width=70)
    
    print("\n✅ All done! Files now have proper line breaks and formatting for screenshots.")
    print("💡 The '*_formatted.json' files are optimized for readability.")