import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\nijjohnson\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedPage:
    """Represents a processed page with OCR results and metadata"""
    page_id: str
    book_id: str
    page_number: int
    text_content: str
    confidence_score: float
    has_images: bool
    image_regions: List[Tuple[int, int, int, int]]  # (x, y, width, height)
    file_path: str
    processed_image_path: Optional[str] = None

class ImagePreprocessor:
    """Handles image preprocessing for better OCR results"""
    
    def __init__(self):
        self.lock = threading.Lock()
    
    def enhance_image(self, image_path: str) -> np.ndarray:
        """Apply image enhancement techniques for better OCR"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            # thresh = cv2.adaptiveThreshold(
            #     denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            #     cv2.THRESH_BINARY, 11, 2
            # )
            
            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            
            # Deskew if needed
            cleaned = self._deskew_image(cleaned)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error enhancing image {image_path}: {e}")
            # Return original grayscale as fallback
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            return img if img is not None else np.zeros((100, 100), dtype=np.uint8)
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct skew in scanned images"""
        try:
            # Find contours
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (presumably the page)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get minimum area rectangle
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]
                
                # Correct angle
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90
                
                # Only correct if angle is significant
                if abs(angle) > 0.5:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return image
            
        except Exception as e:
            logger.warning(f"Could not deskew image: {e}")
            return image
    
    def detect_image_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions that contain images/diagrams vs text"""
        try:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                255 - image, connectivity=8
            )
            
            image_regions = []
            h, w = image.shape
            
            for i in range(1, num_labels):  # Skip background
                x, y, width, height, area = stats[i]
                
                # Heuristics to identify image regions
                aspect_ratio = width / height if height > 0 else 0
                density = area / (width * height) if width * height > 0 else 0
                
                # Large, dense regions with reasonable aspect ratios are likely images
                if (area > 5000 and density > 0.3 and 
                    0.1 < aspect_ratio < 10 and 
                    width > 50 and height > 50):
                    image_regions.append((x, y, width, height))
            
            return image_regions
            
        except Exception as e:
            logger.warning(f"Could not detect image regions: {e}")
            return []

class OCRProcessor:
    """Handles OCR processing with confidence scoring"""
    
    def __init__(self, languages: List[str] = ['eng']):
        self.languages = '+'.join(languages)
        self.config = '--oem 3 --psm 6'  # Use LSTM OCR Engine Mode, uniform block of text
    
    def extract_text_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text and calculate confidence score"""
        try:
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image, 
                lang=self.languages,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text_parts = []
            confidences = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = data['conf'][i]
                
                if text and conf > 0:  # Only include confident text
                    text_parts.append(text)
                    confidences.append(conf)
            
            # Calculate overall confidence
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Join text with appropriate spacing
            full_text = ' '.join(text_parts)
            
            return full_text, avg_confidence
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return "", 0.0
    
    def extract_text_with_layout(self, image: np.ndarray) -> Dict:
        """Extract text while preserving layout information"""
        try:
            # Use different PSM for layout preservation
            layout_config = '--oem 3 --psm 3'  # Fully automatic page segmentation
            
            data = pytesseract.image_to_data(
                image,
                lang=self.languages,
                config=layout_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Group text by blocks and paragraphs
            blocks = {}
            current_block = None
            current_para = None
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text:
                    continue
                
                block_num = data['block_num'][i]
                para_num = data['par_num'][i]
                
                if block_num not in blocks:
                    blocks[block_num] = {'paragraphs': {}, 'bbox': None}
                
                if para_num not in blocks[block_num]['paragraphs']:
                    blocks[block_num]['paragraphs'][para_num] = {
                        'text': [],
                        'bbox': None,
                        'confidence': []
                    }
                
                blocks[block_num]['paragraphs'][para_num]['text'].append(text)
                blocks[block_num]['paragraphs'][para_num]['confidence'].append(data['conf'][i])
            
            # Clean up and format
            formatted_blocks = []
            for block_id, block_data in blocks.items():
                block_text = []
                for para_id, para_data in block_data['paragraphs'].items():
                    para_text = ' '.join(para_data['text'])
                    if para_text.strip():
                        block_text.append(para_text)
                
                if block_text:
                    formatted_blocks.append('\n'.join(block_text))
            
            return {
                'text': '\n\n'.join(formatted_blocks),
                'blocks': blocks
            }
            
        except Exception as e:
            logger.error(f"Layout OCR processing failed: {e}")
            return {'text': '', 'blocks': {}}

class DocumentProcessor:
    """Main document processing orchestrator"""
    
    def __init__(self, output_dir: str = "processed_docs", max_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        self.ocr_processor = OCRProcessor()
        
        # Create subdirectories
        (self.output_dir / "processed_images").mkdir(exist_ok=True)
        (self.output_dir / "text_content").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
    
    def process_book(self, book_path: str, book_id: Optional[str] = None) -> List[ProcessedPage]:
        """Process an entire book (directory of image files)"""
        book_path = Path(book_path)
        
        if book_id is None:
            book_id = book_path.name

        # Recognize multiple image file types
        image_extensions = ["*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(book_path.glob(ext))
        image_files = sorted(set(image_files))  # Remove duplicates and sort

        if not image_files:
            logger.warning(f"No image files found in {book_path}")
            return []
        
        logger.info(f"Processing {len(image_files)} pages for book: {book_id}")
        
        # Process pages in parallel
        processed_pages = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all pages for processing
            future_to_page = {
                executor.submit(self._process_single_page, str(image_file), book_id, i): i
                for i, image_file in enumerate(image_files)
            }
            
            # Collect results
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    processed_page = future.result()
                    if processed_page:
                        processed_pages.append(processed_page)
                        logger.info(f"Processed page {page_num + 1}/{len(image_files)}")
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
        
        # Sort by page number
        processed_pages.sort(key=lambda x: x.page_number)
        
        # Save book metadata
        self._save_book_metadata(book_id, processed_pages)
        
        logger.info(f"Completed processing book {book_id}: {len(processed_pages)} pages")
        return processed_pages
    
    def _process_single_page(self, tiff_path: str, book_id: str, page_number: int) -> Optional[ProcessedPage]:
        """Process a single page"""
        try:
            # Generate page ID
            page_id = f"{book_id}_page_{page_number:04d}"
            
            # Enhance image
            enhanced_image = self.preprocessor.enhance_image(tiff_path)
            
            # Save enhanced image
            processed_image_path = self.output_dir / "processed_images" / f"{page_id}.png"
            cv2.imwrite(str(processed_image_path), enhanced_image)
            
            # Detect image regions
            image_regions = self.preprocessor.detect_image_regions(enhanced_image)
            
            # Extract text
            text_content, confidence = self.ocr_processor.extract_text_with_confidence(enhanced_image)
            
            # Create processed page object
            processed_page = ProcessedPage(
                page_id=page_id,
                book_id=book_id,
                page_number=page_number,
                text_content=text_content,
                confidence_score=confidence,
                has_images=len(image_regions) > 0,
                image_regions=image_regions,
                file_path=tiff_path,
                processed_image_path=str(processed_image_path)
            )
            
            # Save text content
            text_file = self.output_dir / "text_content" / f"{page_id}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Save page metadata
            metadata_file = self.output_dir / "metadata" / f"{page_id}.json"
            metadata = {
                'page_id': page_id,
                'book_id': book_id,
                'page_number': page_number,
                'confidence_score': confidence,
                'has_images': len(image_regions) > 0,
                'image_regions': image_regions,
                'original_file': tiff_path,
                'processed_image': str(processed_image_path),
                'text_length': len(text_content),
                'word_count': len(text_content.split())
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            return processed_page
            
        except Exception as e:
            logger.error(f"Error processing page {tiff_path}: {e}")
            return None
    
    def _save_book_metadata(self, book_id: str, processed_pages: List[ProcessedPage]):
        """Save consolidated book metadata"""
        metadata_file = self.output_dir / "metadata" / f"{book_id}_book_metadata.json"
        
        metadata = {
            'book_id': book_id,
            'total_pages': len(processed_pages),
            'average_confidence': np.mean([p.confidence_score for p in processed_pages]),
            'pages_with_images': sum(1 for p in processed_pages if p.has_images),
            'total_words': sum(len(p.text_content.split()) for p in processed_pages),
            'processing_timestamp': str(np.datetime64('now')),
            'page_list': [p.page_id for p in processed_pages]
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process a book of TIFF files for OCR and image extraction.")
    parser.add_argument("book_path", help="Path to the directory containing TIFF files of the book")
    parser.add_argument("--book_id", help="Optional book ID (defaults to directory name)", default=None)
    parser.add_argument("--output_dir", help="Directory to store processed results", default="./processed_books")
    args = parser.parse_args()

    # Initialize processor
    processor = DocumentProcessor(output_dir=args.output_dir)

    # Process a book
    processed_pages = processor.process_book(args.book_path, book_id=args.book_id)

    # Print summary
    if processed_pages:
        avg_confidence = np.mean([p.confidence_score for p in processed_pages])
        print(f"Processed {len(processed_pages)} pages")
        print(f"Average OCR confidence: {avg_confidence:.2f}")
        print(f"Pages with images: {sum(1 for p in processed_pages if p.has_images)}")