from sheet_processor import UniversalSheetProcessor

# Your sheet1 content (copy from your sheet1.jpg)
sheet1_content = """
# bodyuT

## 1
- tinh & new  
  Love me, Water me.  
  Share me  

## 2
- tinh & new  
  Love me, Water me.  
  Share me
"""

# Your dates content (copy from your unnamed.jpg)
dates_content = """
2023-01-01
2023-02-01
2023-03-01
2023-04-01
"""

# Initialize processor
processor = UniversalSheetProcessor()

# Option 1: Process just sheet1 to unnamed format
print("=== Unnamed Format Output ===")
result = processor.process_sheet1(sheet1_content, output_format="unnamed")
print(result)

# Option 2: Add dates and get table format
print("\n=== Table Format with Dates ===")
result_with_dates = processor.process_with_dates(sheet1_content, dates_content, output_format="table")
print(result_with_dates)

# Option 3: Save to a file
with open('output.txt', 'w') as f:
    f.write(result)

print("\n✓ Output saved to output.txt")