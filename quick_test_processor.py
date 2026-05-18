import cv2
import numpy as np
import os

def quick_test(image_path):
    """
    Quick test of the sticker processor
    """
    from core.sticker_processor import StickerProcessor
    
    print("Quick Test: Sticker Processor")
    print("=" * 50)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found - {image_path}")
        return
    
    # Create processor
    processor = StickerProcessor(red_remove=True, use_fixed_grid=False)
    
    # Process sheet
    stickers, sticker_data = processor.process_sheet(image_path, output_dir="test_output")
    
    print(f"\n✓ Test complete!")
    print(f"Found {len(stickers)} stickers")
    print(f"Check the 'test_output' folder for results")

if __name__ == "__main__":
    print("Sticker Processor Quick Test")
    print("-" * 50)
    image_path = input("Enter path to sheet image: ").strip().strip('"')
    quick_test(image_path)
