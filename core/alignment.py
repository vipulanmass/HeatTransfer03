# core/alignment.py

import cv2
import numpy as np
from typing import Tuple, Optional

class SheetAligner:
    """
    Align current sheet to golden reference using ORB features
    """
    
    def __init__(self, feature_count: int = 2000):
        self.feature_count = feature_count
        self.orb = cv2.ORB_create(nfeatures=feature_count)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def align(self, current_image: np.ndarray, golden_image: np.ndarray) -> np.ndarray:
        """
        Align current image to golden reference
        
        Args:
            current_image: Production image to align
            golden_image: Reference golden image
            
        Returns:
            Warped current image aligned to golden coordinates
        """
        # Convert to grayscale
        if len(current_image.shape) == 3:
            current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            golden_gray = cv2.cvtColor(golden_image, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_image
            golden_gray = golden_image
        
        # Apply CLAHE for better feature detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        current_gray = clahe.apply(current_gray)
        golden_gray = clahe.apply(golden_gray)
        
        # Detect ORB features
        kp1, des1 = self.orb.detectAndCompute(golden_gray, None)
        kp2, des2 = self.orb.detectAndCompute(current_gray, None)
        
        if des1 is None or des2 is None:
            # Fallback: return original image
            return current_image
        
        # Match features
        matches = self.bf.match(des1, des2)
        
        # Sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Use top matches
        good_matches = matches[:min(100, len(matches))]
        
        if len(good_matches) < 4:
            # Not enough matches, return original
            return current_image
        
        # Extract points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate homography with RANSAC
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return current_image
        
        # Warp current image to golden coordinates
        h, w = golden_gray.shape
        aligned = cv2.warpPerspective(current_image, H, (w, h))
        
        return aligned
    
    def get_transform(self, current_image: np.ndarray, golden_image: np.ndarray) -> Optional[np.ndarray]:
        """Get homography matrix without warping"""
        # Convert to grayscale
        if len(current_image.shape) == 3:
            current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            golden_gray = cv2.cvtColor(golden_image, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_image
            golden_gray = golden_image
        
        # Detect ORB features
        kp1, des1 = self.orb.detectAndCompute(golden_gray, None)
        kp2, des2 = self.orb.detectAndCompute(current_gray, None)
        
        if des1 is None or des2 is None:
            return None
        
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:min(100, len(matches))]
        
        if len(good_matches) < 4:
            return None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        return H