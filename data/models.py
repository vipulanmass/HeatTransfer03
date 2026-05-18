# data/models.py

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np
import json

@dataclass
class Character:
    """Represents a single character on a sticker"""
    id: int
    sticker_id: int
    text: str                      # Manually entered text
    bbox: List[int]                # [x, y, w, h] in golden coordinates
    mask_path: str                 # Path to saved binary mask
    components: int                # Expected connected components
    hu_moments: List[float]        # Precomputed Hu moments (7 values)
    area: int                      # Pixel area of character
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class Sticker:
    """Represents a single sticker on a sheet"""
    id: int
    bbox: List[int]                # [x, y, w, h] in sheet coordinates
    characters: List[Character]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'bbox': self.bbox,
            'characters': [c.to_dict() for c in self.characters]
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        characters = [Character.from_dict(c) for c in data['characters']]
        return cls(
            id=data['id'],
            bbox=data['bbox'],
            characters=characters
        )


@dataclass
class Product:
    """Complete product configuration"""
    product_id: str
    name: str
    golden_image_path: str
    stickers: List[Sticker]
    defect_thresholds: Dict[str, float]
    color_layers: Optional[Dict] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'product_id': self.product_id,
            'name': self.name,
            'golden_image_path': self.golden_image_path,
            'stickers': [s.to_dict() for s in self.stickers],
            'defect_thresholds': self.defect_thresholds,
            'color_layers': self.color_layers,
            'created_date': self.created_date,
            'modified_date': self.modified_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        stickers = [Sticker.from_dict(s) for s in data['stickers']]
        return cls(
            product_id=data['product_id'],
            name=data['name'],
            golden_image_path=data['golden_image_path'],
            stickers=stickers,
            defect_thresholds=data['defect_thresholds'],
            color_layers=data.get('color_layers'),
            created_date=data.get('created_date'),
            modified_date=data.get('modified_date')
        )