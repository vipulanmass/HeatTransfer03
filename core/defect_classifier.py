# core/defect_classifier.py

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Defect:
    """Defect classification result"""
    type: str
    severity: float
    details: str
    character_id: int
    character_text: str
    bbox: Tuple[int, int, int, int]

class DefectClassifier:
    """
    Classify character defects using multiple CV techniques
    """
    
    def __init__(self, 
                 missing_threshold: float = 0.30,
                 smear_threshold: float = 0.10,
                 shape_threshold: float = 0.30):
        self.missing_threshold = missing_threshold
        self.smear_threshold = smear_threshold
        self.shape_threshold = shape_threshold
    
    def classify(self, 
                 golden_char,
                 current_roi: np.ndarray,
                 golden_mask: np.ndarray = None) -> Optional[Defect]:
        """
        Classify a character as good or defective
        
        Args:
            golden_char: Golden character object (with mask_path)
            current_roi: ROI from current image containing the character
            golden_mask: Optional pre-loaded golden mask
            
        Returns:
            Defect object or None if good
        """
        # Load golden mask if not provided
        if golden_mask is None:
            golden_mask = self._load_mask(golden_char.mask_path)
        
        # Preprocess current ROI
        current_bin = self._preprocess_current(current_roi, golden_mask.shape)
        
        # Test 1: Pixel difference analysis
        missing_ratio, extra_ratio = self._pixel_diff(golden_mask, current_bin)
        
        if missing_ratio > self.missing_threshold:
            return Defect(
                type='missing_or_broken',
                severity=missing_ratio,
                details=f'Missing {missing_ratio:.1%} of character',
                character_id=golden_char.id,
                character_text=golden_char.text,
                bbox=golden_char.bbox
            )
        
        if extra_ratio > self.smear_threshold:
            return Defect(
                type='smear_bleeding',
                severity=extra_ratio,
                details=f'Extra ink {extra_ratio:.1%} beyond character',
                character_id=golden_char.id,
                character_text=golden_char.text,
                bbox=golden_char.bbox
            )
        
        # Test 2: Connected components analysis
        component_defect = self._components_analysis(golden_char, current_bin)
        if component_defect:
            return component_defect
        
        # Test 3: Shape analysis for borderline cases
        if 0.1 < missing_ratio < self.missing_threshold:
            shape_diff = self._compare_shapes(golden_mask, current_bin)
            if shape_diff > self.shape_threshold:
                return Defect(
                    type='partial_character',
                    severity=shape_diff,
                    details=f'Shape distortion detected (diff={shape_diff:.2f})',
                    character_id=golden_char.id,
                    character_text=golden_char.text,
                    bbox=golden_char.bbox
                )
        
        return None  # Good character
    
    def _preprocess_current(self, roi: np.ndarray, target_shape: Tuple) -> np.ndarray:
        """Preprocess current ROI to binary mask"""
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Clean with morphology
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Resize to match golden if needed
        if binary.shape != target_shape:
            binary = cv2.resize(binary, (target_shape[1], target_shape[0]))
        
        return binary
    
    def _pixel_diff(self, golden_mask: np.ndarray, current_bin: np.ndarray) -> Tuple[float, float]:
        """Calculate missing and extra pixel ratios"""
        # Ensure same dimensions
        if golden_mask.shape != current_bin.shape:
            current_bin = cv2.resize(current_bin, 
                                     (golden_mask.shape[1], golden_mask.shape[0]))
        
        # Calculate difference
        diff = cv2.absdiff(golden_mask, current_bin)
        missing = cv2.bitwise_and(diff, golden_mask)
        extra = cv2.bitwise_and(diff, current_bin)
        
        total_golden = cv2.countNonZero(golden_mask)
        if total_golden == 0:
            return 1.0, 0.0
        
        missing_ratio = cv2.countNonZero(missing) / total_golden
        extra_ratio = cv2.countNonZero(extra) / total_golden
        
        return missing_ratio, extra_ratio
    
    def _components_analysis(self, golden_char, current_bin: np.ndarray) -> Optional[Defect]:
        """Analyze connected components for structural defects"""
        expected = golden_char.components
        
        # Count components in current binary
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(current_bin, 8)
        actual = num_labels - 1
        
        if actual > expected + 1:
            return Defect(
                type='fragmented_broken',
                severity=actual / expected if expected > 0 else 1.0,
                details=f'Expected {expected} components, got {actual}',
                character_id=golden_char.id,
                character_text=golden_char.text,
                bbox=golden_char.bbox
            )
        
        if actual < expected - 1:
            return Defect(
                type='merged_smeared',
                severity=expected / actual if actual > 0 else 1.0,
                details=f'Expected {expected} components, got {actual}',
                character_id=golden_char.id,
                character_text=golden_char.text,
                bbox=golden_char.bbox
            )
        
        return None
    
    def _compare_shapes(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compare shapes using Hu moments"""
        # Find contours
        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours1 or not contours2:
            return 1.0
        
        # Get largest contour
        contour1 = max(contours1, key=cv2.contourArea)
        contour2 = max(contours2, key=cv2.contourArea)
        
        # Calculate Hu moments
        moments1 = cv2.moments(contour1)
        moments2 = cv2.moments(contour2)
        
        hu1 = cv2.HuMoments(moments1).flatten()
        hu2 = cv2.HuMoments(moments2).flatten()
        
        # Log transform for better comparison
        hu1 = -np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
        hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)
        
        # Euclidean distance
        return np.linalg.norm(hu1 - hu2)
    
    @staticmethod
    def _load_mask(mask_path: str) -> np.ndarray:
        """Load binary mask from file"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask