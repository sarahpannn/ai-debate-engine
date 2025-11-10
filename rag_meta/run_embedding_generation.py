#!/usr/bin/env python3
"""
Simple script to run embedding generation with progress monitoring.
"""
import os
import sys
from generate_embeddings import GeminiEmbeddingGenerator, get_table_configs


def check_environment():
    """Check if all required environment variables are set."""
    required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'GEMINI_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   {var}")
        print("\nPlease set these in your environment and try again.")
        return False
    
    print("✅ All required environment variables are set")
    return True


def get_table_stats(generator, table_configs):
    """Get statistics for each table."""
    print("\n📊 Table Statistics:")
    print("-" * 50)
    
    for config in table_configs:
        try:
            # Get total rows
            total_response = generator.supabase.table(config.name).select(
                config.id_column, count='exact'
            ).execute()
            total_rows = total_response.count if hasattr(total_response, 'count') else 0
            
            # Get rows without embeddings
            no_embedding_response = generator.supabase.table(config.name).select(
                config.id_column, count='exact'
            ).is_(config.embedding_column, "null").execute()
            no_embedding_count = no_embedding_response.count if hasattr(no_embedding_response, 'count') else 0
            
            completed = total_rows - no_embedding_count
            percentage = (completed / total_rows * 100) if total_rows > 0 else 0
            
            print(f"{config.name}:")
            print(f"  Total rows: {total_rows}")
            print(f"  Completed: {completed} ({percentage:.1f}%)")
            print(f"  Remaining: {no_embedding_count}")
            print()
            
        except Exception as e:
            print(f"{config.name}: Error getting stats - {e}")
            print()


def main():
    print("🚀 Starting Embedding Generation")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Initialize generator
    try:
        generator = GeminiEmbeddingGenerator(
            supabase_url=os.getenv('SUPABASE_URL'),
            supabase_key=os.getenv('SUPABASE_SERVICE_KEY'),
            gemini_api_key=os.getenv('GEMINI_API_KEY')
        )
        print("✅ Successfully initialized embedding generator")
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        sys.exit(1)
    
    # Get table configurations
    table_configs = get_table_configs()
    
    # Show initial stats
    get_table_stats(generator, table_configs)
    
    # Ask for confirmation
    response = input("Do you want to proceed with embedding generation? (y/N): ").lower()
    if response != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    # Process tables
    print("\n🔄 Starting embedding generation...")
    print("-" * 50)
    
    try:
        generator.process_all_tables(table_configs)
        print("\n✅ Embedding generation completed!")
        
        # Show final stats
        get_table_stats(generator, table_configs)
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        print("Progress has been saved. You can resume by running this script again.")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()