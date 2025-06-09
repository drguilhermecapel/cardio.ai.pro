"""
CLI interface for avatar generation.
"""

import argparse
import sys
from pathlib import Path

from app.services.avatar_generator import generate_avatar


def main() -> None:
    """Main CLI entry point for avatar generation."""
    parser = argparse.ArgumentParser(
        description="Generate hyper-realistic avatars using Stable Diffusion"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for avatar generation (uses default if not provided)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (default: 30)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Image height in pixels (default: 768)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width in pixels (default: 512)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model identifier (default: runwayml/stable-diffusion-v1-5)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="media/avatars",
        help="Output directory for generated images (default: media/avatars)"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = generate_avatar(
            prompt_override=args.prompt,
            seed=args.seed,
            steps=args.steps,
            height=args.height,
            width=args.width,
            model_name=args.model,
            out_dir=Path(args.output_dir),
        )
        
        print(str(output_path.absolute()))
        
    except Exception as e:
        print(f"Error generating avatar: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
