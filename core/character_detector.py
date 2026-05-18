# core/character_detector.py

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DetectedCharacter:
    """Detection result for a single character"""
    id: int
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray
    components: int
    area: int
    centroid: Tuple[float, float]

class CharacterDetector:
    """
    Pure CV character detection using MSER + morphological operations
    """
    
    def __init__(self, 
                 min_area: int = 50,
                 max_area: int = 5000,
                 min_aspect: float = 0.3,
                 max_aspect: float = 3.0,
                 min_solidity: float = 0.5):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_solidity = min_solidity
    
    def detect(self, image: np.ndarray) -> List[DetectedCharacter]:
        """
        Detect characters in image
        
        Args:
            image: BGR or grayscale image
            
        Returns:
            List of DetectedCharacter objects
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Step 1: MSER detection
        mser = cv2.MSER_create()
        mser.setMinArea(self.min_area)
        mser.setMaxArea(self.max_area)
        mser.setDelta(5)
        mser.setMaxVariation(0.5)
        mser.setMinDiversity(0.2)
        
        regions, _ = mser.detectRegions(gray)
        
        # Step 2: Convert regions to bounding boxes with filtering
        candidates = []
        for region in regions:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(region)
            
            # Skip if too small
            if w < 5 or h < 5:
                continue
            
            # Aspect ratio filter
            aspect = w / h if h > 0 else 0
            if not (self.min_aspect < aspect < self.max_aspect):
                continue
            
            # Area filter
            area = w * h
            if not (self.min_area < area < self.max_area):
                continue
            
            # Solidity check (area / convex hull area)
            hull = cv2.convexHull(region)
            hull_area = cv2.contourArea(hull)
            solidity = cv2.contourArea(region) / hull_area if hull_area > 0 else 0
            if solidity < self.min_solidity:
                continue
            
            candidates.append([x, y, w, h])
        
        # Step 3: Merge overlapping candidates
        merged = self._merge_overlapping(candidates, iou_threshold=0.3)
        
        # Step 4: Sort left-to-right, top-to-bottom (reading order)
        merged.sort(key=lambda b: (b[1] // 50, b[0]))
        
        # Step 5: Extract detailed character data
        characters = []
        for i, (x, y, w, h) in enumerate(merged):
            # Extract ROI
            roi = gray[y:y+h, x:x+w]
            
            # Adaptive threshold for binary mask
            binary = self._adaptive_threshold(roi)
            
            # Ensure binary is the same size as ROI
            if binary.shape != (h, w):
                binary = cv2.resize(binary, (w, h))
            
            # Count connected components
            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            components = num_labels - 1
            
            # Calculate centroid
            if components == 1:
                cx, cy = centroids[1]
            else:
                # For multi-component, use overall bounding box center
                cx, cy = x + w/2, y + h/2
            
            characters.append(DetectedCharacter(
                id=i,
                bbox=(x, y, w, h),
                mask=binary,
                components=components,
                area=w * h,
                centroid=(cx, cy)
            ))
        
        return characters
    
    def _adaptive_threshold(self, roi: np.ndarray) -> np.ndarray:
        """Apply adaptive threshold to ROI"""
        # Gaussian adaptive threshold works well for screen printing
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Clean up with morphology
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def _merge_overlapping(self, bboxes: List, iou_threshold: float = 0.3) -> List:
        """Merge bounding boxes that significantly overlap"""
        if not bboxes:
            return []
        
        merged = []
        used = [False] * len(bboxes)
        
        for i, box1 in enumerate(bboxes):
            if used[i]:
                continue
            
            current = list(box1)
            for j, box2 in enumerate(bboxes):
                if i != j and not used[j]:
                    if self._iou(current, box2) > iou_threshold:
                        # Merge boxes
                        x1 = min(current[0], box2[0])
                        y1 = min(current[1], box2[1])
                        x2 = max(current[0] + current[2], box2[0] + box2[2])
                        y2 = max(current[1] + current[3], box2[1] + box2[3])
                        current = [x1, y1, x2 - x1, y2 - y1]
                        used[j] = True
            
            merged.append(current)
            used[i] = True
        
        return merged
    
    @staticmethod
    def _iou(box1, box2) -> float:
        """Calculate Intersection over Union"""
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