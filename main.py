"""
main.py

Entry point for the scene indexing pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, DEFAULT_CONFIG
from indexing.pipeline import IndexingPipeline


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scene-based movie retrieval indexing pipeline"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--clips-dir",
        type=str,
        default=None,
        help="Override: Directory containing movie clip folders"
    )
    
    parser.add_argument(
        "--plot-data",
        type=str,
        default=None,
        help="Override: Path to plot summaries JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override: Directory to save index files"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override: Device to run models on"
    )
    
    parser.add_argument(
        "--vlm-provider",
        type=str,
        default=None,
        choices=["openai", "google"],
        help="Override: VLM provider for captioning"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = DEFAULT_CONFIG
    
    # Apply overrides
    if args.clips_dir:
        config.paths.clips_directory = args.clips_dir
    if args.plot_data:
        config.paths.plot_summaries_path = args.plot_data
    if args.output_dir:
        config.paths.index_directory = args.output_dir
    if args.device:
        config.device = args.device
    if args.vlm_provider:
        config.models.vlm_provider = args.vlm_provider
    
    # Print configuration
    print("Configuration:")
    print(f"  Clips directory: {config.paths.clips_directory}")
    print(f"  Plot summaries: {config.paths.plot_summaries_path}")
    print(f"  Output directory: {config.paths.index_directory}")
    print(f"  Device: {config.device}")
    print(f"  Visual encoder: {config.models.visual_encoder_name}")
    print(f"  Text encoder: {config.models.text_encoder_name}")
    print(f"  Transcriber: Whisper {config.models.whisper_model_size}")
    print(f"  Captioner: {config.models.vlm_provider}/{config.models.vlm_model_name}")
    print()
    
    # Run pipeline
    pipeline = IndexingPipeline(config)
    index = pipeline.run()
    
    print("\nIndexing complete!")
    print(f"Index saved to: {config.paths.index_directory}")
    
    return index


if __name__ == "__main__":
    main()