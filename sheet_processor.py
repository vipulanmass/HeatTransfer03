"""
Universal Sheet Data Extractor & Formatter
Architecture for processing sheet data into standardized output
Inspired by BabyTV reference pattern
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class ProductRecord:
    """Standard product record structure"""
    id: str
    title: str
    tagline: str
    instructions: List[str]
    date_added: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_row(self) -> List[str]:
        """Convert to row format like unnamed.jpg"""
        return [
            self.id,
            self.title,
            self.tagline,
            " | ".join(self.instructions),
            self.date_added or ""
        ]


class SheetParser:
    """
    Base parser for sheet data
    Extracts structured data from various sheet formats
    """
    
    def __init__(self, golden_template: Optional[Dict] = None):
        self.golden_template = golden_template or {
            "pattern": "tiny & new\nLove me, water me.\nShare me",
            "structure": ["title", "tagline", "instructions"]
        }
    
    def parse_sheet1_format(self, content: str) -> List[ProductRecord]:
        """
        Parse the sheet1.jpg format:
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
        records = []
        
        # Split by section markers
        sections = re.split(r'##\s+(\d+)', content)
        
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break
                
            record_id = sections[i].strip()
            record_content = sections[i + 1].strip()
            
            # Parse the content
            lines = [line.strip() for line in record_content.split('\n') if line.strip()]
            
            if lines:
                # Extract title from first line (remove bullet if present)
                title_line = lines[0].lstrip('-').strip()
                
                # Parse tagline and instructions
                tagline = ""
                instructions = []
                
                for line in lines[1:]:
                    if line.startswith('-'):
                        line = line.lstrip('-').strip()
                    
                    # Detect if this is the tagline or instruction
                    if "& new" in line or "tiny" in line:
                        tagline = line
                    else:
                        instructions.append(line)
                
                record = ProductRecord(
                    id=record_id,
                    title=title_line,
                    tagline=tagline,
                    instructions=instructions if instructions else ["Love me, Water me.", "Share me"],
                    date_added=None
                )
                records.append(record)
        
        return records
    
    def parse_unnamed_format(self, content: str) -> List[str]:
        """
        Parse the unnamed.jpg format (date list)
        """
        return [line.strip() for line in content.split('\n') if line.strip()]
    
    def parse_golden_template(self, content: str) -> Dict:
        """
        Parse the BabyTV_reference format
        """
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        return {
            "title": lines[0] if lines else "",
            "tagline": lines[1] if len(lines) > 1 else "",
            "instructions": lines[2:] if len(lines) > 2 else []
        }


class DataTransformer:
    """
    Transform parsed data into desired output formats
    """
    
    def to_unnamed_format(self, records: List[ProductRecord]) -> str:
        """
        Convert product records to unnamed.jpg format (date-like rows)
        """
        rows = []
        for record in records:
            row = f"{record.id} | {record.title} | {record.tagline} | {' | '.join(record.instructions)}"
            rows.append(row)
        return '\n'.join(rows)
    
    def to_table_format(self, records: List[ProductRecord]) -> str:
        """
        Convert to a table format
        """
        if not records:
            return ""
        
        # Create header
        headers = ["ID", "Title", "Tagline", "Instructions", "Date"]
        col_widths = [len(h) for h in headers]
        
        # Calculate column widths
        for record in records:
            col_widths[0] = max(col_widths[0], len(record.id))
            col_widths[1] = max(col_widths[1], len(record.title))
            col_widths[2] = max(col_widths[2], len(record.tagline))
            col_widths[3] = max(col_widths[3], len(" | ".join(record.instructions)))
            col_widths[4] = max(col_widths[4], len(record.date_added or ""))
        
        # Create separator
        separator = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
        
        # Create header row
        header_row = "|"
        for i, header in enumerate(headers):
            header_row += f" {header:<{col_widths[i]}} |"
        
        # Create rows
        rows = [separator, header_row, separator]
        
        for record in records:
            row = "|"
            row += f" {record.id:<{col_widths[0]}} |"
            row += f" {record.title:<{col_widths[1]}} |"
            row += f" {record.tagline:<{col_widths[2]}} |"
            row += f" {' | '.join(record.instructions):<{col_widths[3]}} |"
            row += f" {(record.date_added or ''):<{col_widths[4]}} |"
            rows.append(row)
        
        rows.append(separator)
        
        return '\n'.join(rows)
    
    def to_json_format(self, records: List[ProductRecord]) -> str:
        """Convert to JSON format"""
        return json.dumps([r.to_dict() for r in records], indent=2)


class UniversalSheetProcessor:
    """
    Main processor that orchestrates parsing and transformation
    """
    
    def __init__(self):
        self.parser = SheetParser()
        self.transformer = DataTransformer()
    
    def process_sheet1(self, content: str, output_format: str = "unnamed") -> str:
        """
        Process sheet1 format and output in specified format
        
        Args:
            content: Raw sheet content
            output_format: "unnamed", "table", "json", or "raw"
        
        Returns:
            Formatted output string
        """
        records = self.parser.parse_sheet1_format(content)
        
        if output_format == "unnamed":
            return self.transformer.to_unnamed_format(records)
        elif output_format == "table":
            return self.transformer.to_table_format(records)
        elif output_format == "json":
            return self.transformer.to_json_format(records)
        elif output_format == "raw":
            return str([r.to_dict() for r in records])
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def add_dates_to_records(self, records: List[ProductRecord], dates: List[str]) -> List[ProductRecord]:
        """
        Add dates to records (like unnamed.jpg pattern)
        """
        for i, record in enumerate(records):
            if i < len(dates):
                record.date_added = dates[i]
        return records
    
    def process_with_dates(self, sheet_content: str, dates_content: str, output_format: str = "table") -> str:
        """
        Process sheet with dates from unnamed.jpg format
        """
        records = self.parser.parse_sheet1_format(sheet_content)
        dates = self.parser.parse_unnamed_format(dates_content)
        records = self.add_dates_to_records(records, dates)
        
        if output_format == "table":
            return self.transformer.to_table_format(records)
        elif output_format == "json":
            return self.transformer.to_json_format(records)
        else:
            return self.transformer.to_unnamed_format(records)


# Example usage and testing
def demo():
    """
    Demonstration of the architecture
    """
    
    # Sample sheet1 content
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
    
    # Sample dates content (like unnamed.jpg)
    dates_content = """
2023-01-01
2023-02-01
"""
    
    # Initialize processor
    processor = UniversalSheetProcessor()
    
    print("=" * 60)
    print("OUTPUT LIKE UNNAMED.JPG FORMAT")
    print("=" * 60)
    
    # Process sheet1 to unnamed format
    unnamed_output = processor.process_sheet1(sheet1_content, output_format="unnamed")
    print(unnamed_output)
    
    print("\n" + "=" * 60)
    print("TABLE FORMAT WITH DATES")
    print("=" * 60)
    
    # Process with dates
    table_output = processor.process_with_dates(sheet1_content, dates_content, output_format="table")
    print(table_output)
    
    print("\n" + "=" * 60)
    print("JSON FORMAT")
    print("=" * 60)
    
    # JSON output
    json_output = processor.process_sheet1(sheet1_content, output_format="json")
    print(json_output)


if __name__ == "__main__":
    demo()