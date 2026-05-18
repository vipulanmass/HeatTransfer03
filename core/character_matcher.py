# core/character_matcher.py

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CharacterMatch:
    """Result of matching a golden character to a detected character"""
    golden_id: int
    detected_id: int
    iou: float
    golden_bbox: Tuple[int, int, int, int]
    detected_bbox: Tuple[int, int, int, int]

class CharacterMatcher:
    """
    Match detected characters to golden reference using IoU
    """
    
    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
    
    def match(self, 
              golden_chars: List,
              detected_chars: List) -> Tuple[List[CharacterMatch], List, List]:
        """
        Match golden characters to detected characters
        
        Args:
            golden_chars: List of golden characters (with bbox attribute)
            detected_chars: List of detected characters (with bbox attribute)
            
        Returns:
            matches: List of CharacterMatch objects
            missing_golden: List of unmatched golden characters
            extra_detected: List of unmatched detected characters
        """
        n_golden = len(golden_chars)
        n_detected = len(detected_chars)
        
        if n_golden == 0:
            return [], [], detected_chars
        if n_detected == 0:
            return [], golden_chars, []
        
        # Build cost matrix (1 - IoU)
        cost_matrix = np.zeros((n_golden, n_detected))
        iou_matrix = np.zeros((n_golden, n_detected))
        
        for i, g in enumerate(golden_chars):
            for j, d in enumerate(detected_chars):
                iou = self._compute_iou(g.bbox, d.bbox)
                iou_matrix[i, j] = iou
                cost_matrix[i, j] = 1 - iou
        
        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Build matches with IoU threshold
        matches = []
        used_golden = set()
        used_detected = set()
        
        for r, c in zip(row_indices, col_indices):
            iou = iou_matrix[r, c]
            if iou >= self.iou_threshold:
                matches.append(CharacterMatch(
                    golden_id=golden_chars[r].id,
                    detected_id=detected_chars[c].id,
                    iou=iou,
                    golden_bbox=golden_chars[r].bbox,
                    detected_bbox=detected_chars[c].bbox
                ))
                used_golden.add(r)
                used_detected.add(c)
        
        # Identify missing and extra
        missing = [golden_chars[i] for i in range(n_golden) 
                   if i not in used_golden]
        extra = [detected_chars[i] for i in range(n_detected) 
                 if i not in used_detected]
        
        return matches, missing, extra
    
    @staticmethod
    def _compute_iou(box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes [x, y, w, h]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        
        return intersection / (area1 + area2 - intersection)