#!/usr/bin/env python3
"""Create a looping GIF from all images in a directory."""

import argparse
import sys
from pathlib import Path
from PIL import Image


def create_gif(source_dir, output_path=None, duration_ms=1000, limit=None):
    """
    Create an animated GIF from all images in a directory.
    
    Args:
        source_dir: Path to directory containing images
        output_path: Path to save the GIF (default: animated.gif in source_dir)
        duration_ms: Duration per frame in milliseconds (default: 1000 = 1 second)
        limit: Maximum number of images to include (default: None = all images)
    """
    source_path = Path(source_dir)
    
    if not source_path.is_dir():
        print(f"Error: {source_dir} is not a valid directory")
        sys.exit(1)
    
    # Supported image formats
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    
    # Collect all image files
    image_files = sorted([
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])
    
    # Limit number of images if requested
    if limit is not None:
        image_files = image_files[:limit]
    
    if not image_files:
        print(f"Error: No image files found in {source_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images")
    
    # Load images
    images = []
    for i, img_path in enumerate(image_files, 1):
        try:
            img = Image.open(img_path).convert('RGB')
            images.append(img)
            print(f"  [{i}/{len(image_files)}] Loaded {img_path.name} ({img.size})")
        except Exception as e:
            print(f"  Warning: Could not load {img_path.name}: {e}")
    
    if not images:
        print("Error: Could not load any images")
        sys.exit(1)
    
    # Determine output path
    if output_path is None:
        output_path = source_path / "animated.gif"
    else:
        output_path = Path(output_path)
    
    # Create GIF
    print(f"\nCreating GIF with {duration_ms}ms per frame...")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,  # 0 = infinite loop
        optimize=False
    )
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ GIF saved to {output_path} ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Create a looping GIF from all images in a directory"
    )
    parser.add_argument(
        "directory",
        help="Directory containing images"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output GIF path (default: animated.gif in source directory)"
    )
    parser.add_argument(
        "-d", "--duration",
        type=int,
        default=1000,
        help="Duration per frame in milliseconds (default: 1000)"
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Maximum number of images to include (default: all)"
    )
    
    args = parser.parse_args()
    
    create_gif(args.directory, args.output, args.duration, args.limit)


if __name__ == "__main__":
    main()
