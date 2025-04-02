#!/usr/bin/env python3

import sys
import os
import cv2
import torch
import numpy as np
import urllib.request
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QPushButton, 
                            QFrame, QSizePolicy, QFileDialog, QProgressBar, QSlider, QCheckBox, QDialog, QSplitter, QScrollArea, QGridLayout)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QSize, QThread, pyqtSignal, pyqtProperty, QRect, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QFont, QDragEnterEvent, QDropEvent, QPainter, QPainterPath, QColor, QPen, QFontMetrics
from PyQt6.QtSvg import QSvgRenderer

# Import YOLO from ultralytics
from ultralytics import YOLO

import pyqtgraph as pg
from collections import deque
import time

# Constants for styling
DARK_BG_COLOR = "#1E1E1E"
PANEL_BG_COLOR = "#252526"
WIDGET_BG_COLOR = "#2D2D30"
DROPDOWN_BG_COLOR = "#333337"
BORDER_COLOR = "#3E3E42"
TEXT_COLOR = "#E0E0E0"
MUTED_TEXT_COLOR = "#AAAAAA"
ACCENT_COLOR = "#007ACC"
LIGHTER_ACCENT_COLOR = "#1C97EA"
DARKER_ACCENT_COLOR = "#005A9C"
GRID_COLOR = (80, 80, 80)  # For OpenCV which uses RGB tuples

# Font styles
DEFAULT_FONT = "font-family: Arial;"
HEADER_FONT_STYLE = f"{DEFAULT_FONT} font-size: 16px; font-weight: bold; color: {TEXT_COLOR}; border: none;"
SUBHEADER_FONT_STYLE = f"{DEFAULT_FONT} font-size: 14px; color: {TEXT_COLOR}; border: none;"
VALUE_FONT_STYLE = f"{DEFAULT_FONT} font-size: 14px; font-weight: bold; color: {ACCENT_COLOR}; border: none;"
LARGE_VALUE_FONT_STYLE = f"{DEFAULT_FONT} font-size: 48px; font-weight: bold; color: {ACCENT_COLOR}; border: none;"

# Button style templates
BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {DARKER_ACCENT_COLOR};
        color: white;
        border: 1px solid {ACCENT_COLOR};
        border-radius: 3px;
        {DEFAULT_FONT}
        font-size: 13px;
        padding: 0px;
        min-width: 32px;
        min-height: 32px;
        max-width: 32px;
        max-height: 32px;
        margin: 0px;
        text-align: center;
    }}
    
    QPushButton:hover {{
        background-color: {ACCENT_COLOR};
        border: 1px solid {LIGHTER_ACCENT_COLOR};
    }}
    
    QPushButton:pressed {{
        background-color: {DARKER_ACCENT_COLOR};
        border: 1px solid {ACCENT_COLOR};
    }}
    
    QPushButton:checked {{
        background-color: {DARKER_ACCENT_COLOR};
        border: 1px solid {ACCENT_COLOR};
    }}
    
    QPushButton:disabled {{
        background-color: #3E3E3E;
        color: #888888;
        border: 1px solid #505050;
    }}
"""

EXPORT_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {DARKER_ACCENT_COLOR};
        color: white;
        border: 1px solid {ACCENT_COLOR};
        border-radius: 3px;
        {DEFAULT_FONT}
        font-size: 13px;
        padding: 6px 12px;
    }}
    
    QPushButton:hover {{
        background-color: {ACCENT_COLOR};
        border: 1px solid {LIGHTER_ACCENT_COLOR};
    }}
    
    QPushButton:pressed {{
        background-color: {DARKER_ACCENT_COLOR};
        border: 1px solid {ACCENT_COLOR};
    }}
    
    QPushButton:disabled {{
        background-color: #3E3E3E;
        color: #888888;
        border: 1px solid #505050;
    }}
"""

class ToggleSwitch(QWidget):
    """Modern toggle switch widget"""
    
    toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(50, 24)
        self.setMaximumHeight(24)
        
        # State
        self.checked = False
        self.thumb_position = 2  # Initial position
        
        # Set cursor to pointing hand 
        self.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def paintEvent(self, event):
        """Custom paint event to draw the toggle switch"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get dimensions
        width = self.width()
        height = self.height()
        
        # Determine colors based on state and enabled status
        if not self.isEnabled():
            # Disabled state
            track_color = QColor('#3E3E42')  # Gray for disabled
            thumb_color = QColor('#888888')  # Light gray for disabled thumb
        else:
            # Enabled state
            track_color = QColor(ACCENT_COLOR) if self.checked else QColor(BORDER_COLOR)
            thumb_color = QColor('#FFFFFF') if self.checked else QColor('#AAAAAA')
        
        # Draw track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        track_rect = QRect(0, 0, width, height)
        painter.drawRoundedRect(track_rect, height//2, height//2)
        
        # Calculate thumb position
        thumb_width = height - 4
        if self.checked:
            thumb_pos = width - thumb_width - 2
        else:
            thumb_pos = 2
        
        # Draw thumb
        painter.setBrush(thumb_color)
        thumb_rect = QRect(thumb_pos, 2, thumb_width, thumb_width)
        painter.drawEllipse(thumb_rect)
    
    def mousePressEvent(self, event):
        """Handle mouse press events to toggle the switch"""
        if not self.isEnabled():
            return  # Do nothing if disabled
            
        if event.button() == Qt.MouseButton.LeftButton:
            self.checked = not self.checked
            self.update()  # Force redraw
            self.toggled.emit(self.checked)
            event.accept()
    
    def setChecked(self, checked):
        """Set the checked state programmatically"""
        if self.checked != checked:
            self.checked = checked
            self.update()  # Force redraw
    
    def isChecked(self):
        """Return the current checked state"""
        return self.checked

class ModernBoxedSlider(QWidget):
    """Custom slider widget that looks like a filled progress bar with text inside"""
    
    valueChanged = pyqtSignal(int)
    
    def __init__(self, integer_display=False, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 32)  # Match dropdown height
        self.setMaximumHeight(32)     # Match dropdown height
        
        # Default slider properties
        self.minimum = 10
        self.maximum = 90
        self.value = 40
        self.pressed = False
        self.hover = False
        self.integer_display = integer_display  # New property to control display format
        
        # Set cursor to pointing hand
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
    
    def setValue(self, value):
        """Set slider value and emit change signal if needed"""
        value = max(self.minimum, min(self.maximum, value))
        if self.value != value:
            self.value = value
            self.update()  # Trigger repaint
            self.valueChanged.emit(value)
    
    def getValue(self):
        """Get current slider value"""
        return self.value
    
    def setRange(self, minimum, maximum):
        """Set slider range"""
        self.minimum = minimum
        self.maximum = maximum
        self.value = max(self.minimum, min(self.maximum, self.value))
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event to draw the slider"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get dimensions
        width = self.width()
        height = self.height()
        
        # Calculate filled width based on value
        value_range = self.maximum - self.minimum
        value_position = (self.value - self.minimum) / value_range if value_range > 0 else 0
        filled_width = int(width * value_position)
        
        # Draw background (unfilled part) - using dropdown background color
        painter.setPen(QPen(QColor(BORDER_COLOR), 1))  # Match dropdown border
        painter.setBrush(QColor(DROPDOWN_BG_COLOR))  # Match dropdown background
        painter.drawRoundedRect(0, 0, width, height, 4, 4)  # Match dropdown border radius
        
        # Draw filled part
        if filled_width > 0:
            # Use lighter blue when hovered
            fill_color = QColor(ACCENT_COLOR) if self.hover else QColor(DARKER_ACCENT_COLOR)
            painter.setBrush(fill_color)
            
            # Set the border color for filled part to lighter accent color
            painter.setPen(QPen(QColor(LIGHTER_ACCENT_COLOR), 0.5))
            
            # Create a rectangular path for the filled portion
            filled_rect = QRect(0, 0, filled_width, height)
            
            # Need to handle the rounded corners specially to match other elements
            if filled_width < width:
                # If not fully filled, use a clipped path to draw
                path = QPainterPath()
                path.addRoundedRect(0, 0, width, height, 4, 4)
                painter.setClipPath(path)
                painter.drawRect(filled_rect)
                painter.setClipping(False)
            else:
                # If fully filled, draw with rounded corners
                painter.drawRoundedRect(0, 0, filled_width, height, 4, 4)
        
        # Draw border again to ensure it's visible
        border_color = ACCENT_COLOR if self.hover else BORDER_COLOR
        painter.setPen(QPen(QColor(border_color), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(0, 0, width, height, 5, 5)
        
        # Format the value text based on the display mode
        if self.integer_display:
            value_text = str(self.value)  # Just show the integer value
        else:
            value_text = f"{self.value / 100:.2f}"  # Show as decimal (original behavior)
        
        # Draw text - matching dropdown font style
        painter.setPen(QColor(TEXT_COLOR))  # Match dropdown text color
        font = QFont('Segoe UI', 14)  # Match dropdown font
        painter.setFont(font)
        text_rect = QRect(0, 0, width, height)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, value_text)
    
    def mousePressEvent(self, event):
        """Handle mouse press event"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed = True
            # Handle both PyQt5 and PyQt6 mouse event styles
            try:
                # PyQt6 style
                x = event.position().x()
            except:
                # PyQt5 compatibility
                x = event.x()
            self.updateValueFromMouse(x)
            self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move event"""
        # Handle both PyQt5 and PyQt6 mouse event styles
        try:
            # PyQt6 style
            x = event.position().x()
        except:
            # PyQt5 compatibility
            x = event.x()
        
        # Update hover state
        self.hover = True
        self.update()
        
        if self.pressed:
            self.updateValueFromMouse(x)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release event"""
        if event.button() == Qt.MouseButton.LeftButton and self.pressed:
            self.pressed = False
            # Handle both PyQt5 and PyQt6 mouse event styles
            try:
                # PyQt6 style
                x = event.position().x()
            except:
                # PyQt5 compatibility
                x = event.x()
            self.updateValueFromMouse(x)
            self.update()
    
    def leaveEvent(self, event):
        """Handle mouse leave event"""
        self.hover = False
        self.update()
    
    def updateValueFromMouse(self, x):
        """Update slider value based on mouse position"""
        value_range = self.maximum - self.minimum
        value_position = max(0, min(1, x / self.width()))
        new_value = self.minimum + int(value_position * value_range)
        self.setValue(new_value)
    
    def sizeHint(self):
        """Provide a default size hint"""
        return QSize(150, 32)

class ModelDownloadThread(QThread):
    """Thread for downloading YOLO models"""
    progress_update = pyqtSignal(int, str)  # Progress percentage, message
    download_complete = pyqtSignal(bool, str)  # Success, model path
    
    def __init__(self, model_name, model_url, save_path):
        super().__init__()
        self.model_name = model_name
        self.model_url = model_url
        self.save_path = save_path
        
    def run(self):
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            
            # Download with progress reporting
            self.download_with_progress(self.model_url, self.save_path)
            
            self.progress_update.emit(100, f"Model {self.model_name} downloaded successfully")
            self.download_complete.emit(True, self.save_path)
            
        except Exception as e:
            error_msg = f"Error downloading model {self.model_name}: {str(e)}"
            print(error_msg)
            self.progress_update.emit(0, error_msg)
            self.download_complete.emit(False, "")
    
    def download_with_progress(self, url, save_path):
        """Download a file with progress reporting"""
        
        def progress_callback(count, block_size, total_size):
            # Calculate progress percentage
            if total_size > 0:
                percentage = min(int(count * block_size * 100 / total_size), 100)
                self.progress_update.emit(percentage, f"Downloading {self.model_name}: {percentage}%")
        
        # Download the file
        self.progress_update.emit(0, f"Starting download of {self.model_name}...")
        urllib.request.urlretrieve(url, save_path, progress_callback)

class VideoFrameThread(QThread):
    """Separate thread for handling video frames to prevent UI slowdowns"""
    frame_ready = pyqtSignal(object)
    video_ended = pyqtSignal()  # Signal when video reaches end - at class level
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.paused = False
        self.loop_detected = False  # Flag to indicate video has looped

    def set_capture(self, cap):
        self.cap = cap
    
    def stop(self):
        self.running = False
        self.wait()
    
    def pause(self, paused):
        self.paused = paused
    
    def run(self):
        self.running = True
        
        # For local videos or webcams
        while self.running and self.cap is not None and self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                else:
                    # Video ended - don't automatically restart
                    # Just emit the end-of-video signal
                    self.video_ended.emit()
            
            # Sleep to control frame rate
            self.msleep(30)  # ~33 fps

class YoloDetectionThread(QThread):
    """Separate thread for YOLO detection to prevent UI slowdowns"""
    detection_ready = pyqtSignal(object, int, list)  # Frame, count, boxes
    model_loaded = pyqtSignal(bool, str)  # Success, message
    
    def __init__(self, model_path="yolov8n.pt"):
        super().__init__()
        self.frame_queue = []
        self.running = False
        self.model = None
        self.model_path = model_path
        self.processing = False
        self.loading_model = False
        self.confidence_threshold = 0.4  # Default threshold
        
    def set_model_path(self, model_path):
        """Set a new model path and reset the model"""
        self.model_path = model_path
        self.model = None
        
    def add_frame(self, frame):
        if frame is not None and not self.processing:
            self.frame_queue = [frame.copy()]  # Only keep the latest frame
    
    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold for detections"""
        self.confidence_threshold = threshold
    
    def stop(self):
        self.running = False
        self.wait()
    
    def load_model(self):
        """Load YOLO model"""
        if self.loading_model:
            return
            
        self.loading_model = True
        
        try:
            self.model = YOLO(self.model_path)
            self.model_loaded.emit(True, f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            error_msg = f"Error loading YOLO model: {e}"
            self.model_loaded.emit(False, error_msg)
        
        self.loading_model = False
    
    def run(self):
        self.running = True
        
        # Load YOLO model if not already loaded
        if self.model is None:
            self.load_model()
        
        while self.running:
            if len(self.frame_queue) > 0 and self.model is not None:
                self.processing = True
                frame = self.frame_queue.pop(0)
                
                try:
                    # Run YOLO detection on the frame
                    results = self.model(frame, classes=0)  # Class 0 is 'person' in COCO dataset
                    
                    # Collect people boxes for heatmap and count
                    people_count = 0
                    boxes = []
                    
                    for result in results:
                        result_boxes = result.boxes
                        for box in result_boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            confidence = float(box.conf[0])
                            
                            # Only count if confidence is above threshold
                            if confidence > self.confidence_threshold:
                                # Draw bounding box
                                color = (0, 255, 0)  # Green for people
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Add confidence text
                                conf_text = f"{confidence:.2f}"
                                cv2.putText(frame, conf_text, (x1, y1-5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                # Store box coordinates for heatmap
                                boxes.append((x1, y1, x2, y2))
                                
                                # Increment people count
                                people_count += 1
                    
                    # Emit the processed frame, people count, and boxes for heatmap
                    self.detection_ready.emit(frame, people_count, boxes)
                    
                except Exception as e:
                    print(f"Error in YOLO detection: {e}")
                
                self.processing = False
            
            # Sleep to prevent high CPU usage
            self.msleep(10)

class DragDropVideoLabel(QLabel):
    """Custom QLabel with drag and drop functionality for videos"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.parent_app = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.is_hovered = False
        
        # Normal styling
        self.normal_style = f"""
            QLabel {{
                {DEFAULT_FONT}
                font-size: 14px;
                color: {MUTED_TEXT_COLOR};
                background-color: {DARK_BG_COLOR};
                border-radius: 4px;
            }}
        """
        
        # Mouse hover styling (without drag)
        self.hover_style = f"""
            QLabel {{
                {DEFAULT_FONT}
                font-size: 14px;
                color: {MUTED_TEXT_COLOR};
                background-color: #1a1a1a;
                border-radius: 4px;
            }}
        """
        
        # Highlight styling for drag hover
        self.highlight_style = f"""
            QLabel {{
                {DEFAULT_FONT}
                font-size: 14px;
                color: #FFFFFF;
                background-color: #1a1a1a;
                border: 2px dashed {ACCENT_COLOR};
                border-radius: 4px;
            }}
        """
        
        self.setStyleSheet(self.normal_style)
        
        # Make cursor a pointer to indicate it's clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Container for the icon and text
        self.content_layout = QVBoxLayout(self)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        self.content_layout.setSpacing(15)
        
        # Icon label for SVG
        self.icon_label = QLabel(self)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setFixedSize(80, 80)
        self.icon_label.setStyleSheet("border: none;")  # Remove borders
        
        # Text label
        self.text_label = QLabel(f"<span style='color:{ACCENT_COLOR};'>(Upload Video)</span><br>Or select a sample source and press play")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet(f"color: {MUTED_TEXT_COLOR}; border: none;")  # Remove borders
        
        # Add widgets to layout
        self.content_layout.addStretch(1)
        self.content_layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(self.text_label, 0, Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addStretch(1)
    
    def set_parent_app(self, app):
        """Set parent application reference to access video loading methods"""
        self.parent_app = app
    
    def set_default_content(self):
        """Set the default content with SVG icon and text"""
        # Clear the video pixmap if present
        self.clear()
        
        # Apply styling based on hover state
        if self.is_hovered:
            self.setStyleSheet(self.hover_style)
        else:
            self.setStyleSheet(self.normal_style)
        
        # Try to load SVG icon from assets directory
        assets_dir = os.path.join(os.getcwd(), "assets")
        svg_path = os.path.join(assets_dir, "video-upload.svg")
        
        if os.path.exists(svg_path):
            try:
                # Create a QPixmap to display the SVG
                svg_renderer = QSvgRenderer(svg_path)
                if svg_renderer.isValid():
                    pixmap = QPixmap(80, 80)
                    pixmap.fill(Qt.GlobalColor.transparent)
                    painter = QPainter(pixmap)
                    svg_renderer.render(painter)
                    painter.end()
                    self.icon_label.setPixmap(pixmap)
                else:
                    # Fallback if renderer isn't valid
                    self.icon_label.setText("📁")
                    self.icon_label.setStyleSheet(f"font-size: 48px; color: {MUTED_TEXT_COLOR}; border: none;")
            except Exception as e:
                # Fallback icon
                self.icon_label.setText("📁")
                self.icon_label.setStyleSheet(f"font-size: 48px; color: {MUTED_TEXT_COLOR}; border: none;")
        else:
            # Fallback icon
            self.icon_label.setText("📁")
            self.icon_label.setStyleSheet(f"font-size: 48px; color: {MUTED_TEXT_COLOR}; border: none;")
            
        # Show the container with icon and text
        self.icon_label.setVisible(True)
        self.text_label.setVisible(True)
    
    def enterEvent(self, event):
        """Handle mouse enter events - darken background"""
        if not self.is_hovered:
            self.is_hovered = True
            self.setStyleSheet(self.hover_style)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave events - restore normal background"""
        if self.is_hovered:
            self.is_hovered = False
            self.setStyleSheet(self.normal_style)
        super().leaveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle click events to open file dialog"""
        if self.parent_app:
            self.parent_app.open_file_dialog()
        super().mouseReleaseEvent(event)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drag enter events that contain file URLs"""
        if event.mimeData().hasUrls():
            # Apply highlight style with border
            self.is_hovered = True
            self.setStyleSheet(self.highlight_style)
            event.acceptProposedAction()
    
    def dragLeaveEvent(self, event):
        """Reset style when drag leaves"""
        self.is_hovered = False
        self.setStyleSheet(self.normal_style)
        super().dragLeaveEvent(event)
    
    def dragMoveEvent(self, event):
        """Accept drag move events for file URLs"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle file drop events"""
        # Reset style
        self.is_hovered = False
        self.setStyleSheet(self.normal_style)
        
        if event.mimeData().hasUrls() and self.parent_app is not None:
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()  # Get the first dropped file path
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
                if any(file_path.lower().endswith(ext) for ext in video_extensions):
                    self.parent_app.load_video_from_path(file_path)
                    event.acceptProposedAction()

class CrowdSenseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize UI
        self.setWindowTitle("CrowdSense")
        self.setMinimumSize(1100, 700)

        # Set application style
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {DARK_BG_COLOR};
                color: {TEXT_COLOR};
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
        """)
        
        # Main widget with padding
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout with updated padding (24px)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(24, 24, 24, 24)
        self.main_layout.setSpacing(16)

        self.export_heatmap_button = None
        self.export_graph_button = None
        
        # Define available YOLO models
        self.available_models = {
            "YOLOv8n (Nano)": {
                "path": "yolov8n.pt",
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                "description": "Smallest and fastest model, best for weaker hardware",
                "size": "6.2 MB"
            },
            "YOLOv8s (Small)": {
                "path": "yolov8s.pt",
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
                "description": "Good balance of speed and accuracy",
                "size": "21.5 MB"
            },
            "YOLOv8m (Medium)": {
                "path": "yolov8m.pt",
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
                "description": "Better accuracy, still reasonable performance",
                "size": "51.5 MB"
            },
            "YOLOv8l (Large)": {
                "path": "yolov8l.pt",
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
                "description": "High accuracy, slower performance",
                "size": "87.5 MB"
            },
            "YOLOv8x (XLarge)": {
                "path": "yolov8x.pt",
                "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
                "description": "Best accuracy, slowest performance",
                "size": "136.5 MB"
            }
        }
        
        # Initialize model to YOLOv8n by default
        self.current_model_key = "YOLOv8n (Nano)"
        self.model_path = self.available_models[self.current_model_key]["path"]
        
        # Directory for storing models
        self.models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Flag to track if model is downloading
        self.model_downloading = False
        self.yolo_ready = False

        # Initialize video capture in a separate thread
        self.cap = None
        self.video_thread = VideoFrameThread()
        self.video_thread.frame_ready.connect(self.process_video_frame)
        
        # Initialize YOLO detection thread
        self.yolo_thread = YoloDetectionThread(self.model_path)
        self.yolo_thread.detection_ready.connect(self.display_detection_results)
        self.yolo_thread.model_loaded.connect(self.on_model_loaded)
        
        # Initialize model download thread (will be created when needed)
        self.download_thread = None
        
        # Frame buffers and detection data
        self.current_frame = None  # Raw current frame
        self.displayed_frame = None  # Processed frame with heatmap (if enabled)
        self.last_detected_boxes = []  # Store the last detected boxes

        # People counting
        self.people_count = 0
        self.people_count_history = deque(maxlen=24)  # For smoothing
        self.smoothed_people_count = 0
        
        # Playback state
        self.paused = False
        self.confidence_threshold = 0.4  # Default value

        # Heatmap properties
        self.heatmap_enabled = False
        self.heatmap_opacity = 0.7
        self.heatmap_accumulator = None
        self.aggregate_heatmap_accumulator = None  # This will store the aggregate heatmap with no decay
        self.aggregate_frame_count = 0  # Track how many frames contributed to aggregate
        self.heatmap_decay = 0.99
        self.heatmap_blur_size = 21
        self.heatmap_radius = 2
        self.heatmap_intensity = 0.6
        self.heatmap_scale_factor = 0.2
        self.heatmap_neighbor_radius = 4

        # Video timer properties
        self.video_time_ms = 0
        self.last_frame_time = 0
        self.frame_interval = 33  # Default frame interval (30 fps)

        self.video_thread.video_ended.connect(self.on_video_ended)

        # Crowd threshold parameters
        self.crowd_detection_enabled = False
        self.crowd_size_threshold = 10       # Default threshold for people count
        self.smoothing_window_size = 24      # Default window size
        self.people_count_history = deque(maxlen=self.smoothing_window_size)
        self.threshold_alert_active = False  # Current alert status
        self.threshold_history = []          # Store alert history with timestamps

        self.peak_count = 0
        self.peak_time_ms = 0
        self.offpeak_count = float('inf')  # Start with infinity so any count will be lower
        self.offpeak_time_ms = 0
        self.peak_marker = None      # Graph marker for peak
        self.offpeak_marker = None   # Graph marker for off-peak

        # Setup UI components
        self.setup_ui()
        
    def setup_ui(self):
        # Header section
        self.setup_header()
        
        # Main content area with video and metrics
        self.setup_main_content()
    
    def setup_header(self):
        header_container = QWidget()
        header_layout = QVBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(16)
        
        # Title and subtitle
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)
        
        title = QLabel("CrowdSense")
        title.setStyleSheet(f"""
            {DEFAULT_FONT}
            font-size: 26px;
            font-weight: bold;
            color: #FFFFFF;
        """)
        
        subtitle = QLabel("A Real-Time Crowd Monitoring Utility")
        subtitle.setStyleSheet(f"""
            {DEFAULT_FONT}
            font-size: 14px;
            color: {MUTED_TEXT_COLOR};
        """)
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        
        # Controls section
        controls_section = QWidget()
        controls_section_layout = QVBoxLayout(controls_section)
        controls_section_layout.setContentsMargins(0, 0, 0, 0)
        controls_section_layout.setSpacing(8)
        
        # First row: Model selection and confidence threshold
        self.create_model_selection_row(controls_section_layout)
        
        # Second row: Video source selection and playback controls
        self.create_source_selection_row(controls_section_layout)
        
        # Add components to header layout
        header_layout.addWidget(title_container)
        header_layout.addWidget(controls_section)
        
        # Add header to main layout
        self.main_layout.addWidget(header_container)
    
    def create_model_selection_row(self, parent_layout):
        first_row_container = QWidget()
        first_row_layout = QHBoxLayout(first_row_container)
        first_row_layout.setContentsMargins(0, 0, 0, 0)
        first_row_layout.setSpacing(16)
        
        # Model selection part
        model_container = QWidget()
        model_layout = QHBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)
        
        model_label = QLabel("Select YOLO Model:")
        model_label.setStyleSheet(SUBHEADER_FONT_STYLE)
        
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.setup_dropdown_style(self.model_combo)
        
        # Populate the model combo box
        for model_name in self.available_models:
            model_info = self.available_models[model_name]
            display_text = f"{model_name} - {model_info['description']} ({model_info['size']})"
            self.model_combo.addItem(display_text, model_name)
        
        # Select the default model
        default_index = list(self.available_models.keys()).index(self.current_model_key)
        self.model_combo.setCurrentIndex(default_index)
        
        # Connect model selection change event
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo, 1)
        
        # Confidence threshold part
        threshold_container = QWidget()
        threshold_layout = QHBoxLayout(threshold_container)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(8)
        
        threshold_label = QLabel("Set Confidence ≥:")
        threshold_label.setStyleSheet(SUBHEADER_FONT_STYLE)
        
        # Create the modern boxed slider
        self.threshold_slider = ModernBoxedSlider()
        self.threshold_slider.setMinimumWidth(100)
        self.threshold_slider.setRange(10, 90)
        self.threshold_slider.setValue(int(self.confidence_threshold * 100))
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider, 1)
        
        # Add both containers to the first row
        first_row_layout.addWidget(model_container, 3)
        first_row_layout.addWidget(threshold_container, 1)
        
        parent_layout.addWidget(first_row_container)
    
    def create_source_selection_row(self, parent_layout):
        source_container = QWidget()
        source_layout = QHBoxLayout(source_container)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(8)
        
        source_label = QLabel("Select Sample Source:")
        source_label.setStyleSheet(SUBHEADER_FONT_STYLE)
        
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(300)
        self.setup_dropdown_style(self.source_combo)
        
        # Populate the combobox with video sources
        self.populate_sources()
        
        # Control buttons container
        buttons_container = self.create_playback_buttons()
        
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_combo, 1)
        source_layout.addWidget(buttons_container)
        
        parent_layout.addWidget(source_container)
    
    def create_playback_buttons(self):
        buttons_container = QWidget()
        buttons_layout = QHBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)
        
        self.restart_button = QPushButton("↻")
        self.restart_button.setToolTip("Restart Video")
        self.restart_button.setStyleSheet(BUTTON_STYLE)
        self.restart_button.clicked.connect(self.restart_video)
        self.restart_button.setEnabled(False)  # Initially disabled
                
        self.play_button = QPushButton("▶")
        self.play_button.setToolTip("Start")
        self.play_button.setStyleSheet(BUTTON_STYLE)
        self.play_button.clicked.connect(self.start_video)
        
        self.pause_button = QPushButton("⏸")
        self.pause_button.setToolTip("Pause")
        self.pause_button.setStyleSheet(BUTTON_STYLE)
        self.pause_button.clicked.connect(self.pause_video)
        self.pause_button.setEnabled(False)
        
        self.stop_button = QPushButton("⏹")
        self.stop_button.setToolTip("Stop")
        self.stop_button.setStyleSheet(BUTTON_STYLE)
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)
        
        # Add buttons to layout
        buttons_layout.addWidget(self.restart_button)
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.stop_button)
        
        return buttons_container

    def create_peak_time_widget(self):
        """Create a simplified widget for displaying peak and off-peak times"""
        peak_time_widget = QWidget()
        peak_time_widget.setStyleSheet(f"""
            background-color: {WIDGET_BG_COLOR};
            border-radius: 4px;
        """)
        
        peak_layout = QVBoxLayout(peak_time_widget)
        peak_layout.setContentsMargins(16, 16, 16, 16)
        peak_layout.setSpacing(12)
        
        # Header for the section - match styling with other section headers
        section_header = QLabel("Traffic Analysis")
        section_header.setStyleSheet(f"""
            font-family: Arial;
            font-size: 14px;
            color: #CCCCCC;
            border: none;
        """)
        peak_layout.addWidget(section_header)
        
        # Container for peak and off-peak rows
        rows_container = QWidget()
        rows_container.setStyleSheet("border: none;")
        rows_layout = QVBoxLayout(rows_container)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(8)  # Space between rows
        
        # Peak time row (horizontal)
        peak_row = QWidget()
        peak_row.setStyleSheet("border: none;")
        peak_row_layout = QHBoxLayout(peak_row)
        peak_row_layout.setContentsMargins(0, 0, 0, 0)
        peak_row_layout.setSpacing(12)  # Space between elements
        
        # Small colored indicator for peak
        peak_indicator = QLabel()
        peak_indicator.setFixedSize(10, 10)
        peak_indicator.setStyleSheet("""
            background-color: #FF5555;
            border-radius: 5px;
            border: none;
        """)
        
        peak_label = QLabel("Peak Time:")
        peak_label.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            font-weight: bold;
            color: #CCCCCC;
            border: none;
        """)
        peak_label.setFixedWidth(100)  # Fixed width for label
        
        # Using app accent color for both time labels
        self.peak_time_value = QLabel("--:--:--")
        self.peak_time_value.setStyleSheet(f"""
            font-family: Arial;
            font-size: 14px;
            font-weight: bold;
            color: {ACCENT_COLOR};
            border: none;
        """)
        self.peak_time_value.setFixedWidth(60)  # Fixed width for time value
        
        self.peak_count_value = QLabel("(0 people)")
        self.peak_count_value.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #AAAAAA;
            border: none;
        """)
        
        peak_row_layout.addWidget(peak_indicator)
        peak_row_layout.addWidget(peak_label)
        peak_row_layout.addWidget(self.peak_time_value)
        peak_row_layout.addWidget(self.peak_count_value)
        peak_row_layout.addStretch(1)  # Push everything to the left
        
        # Off-peak time row (horizontal)
        offpeak_row = QWidget()
        offpeak_row.setStyleSheet("border: none;")
        offpeak_row_layout = QHBoxLayout(offpeak_row)
        offpeak_row_layout.setContentsMargins(0, 0, 0, 0)
        offpeak_row_layout.setSpacing(12)  # Space between elements
        
        # Small colored indicator for off-peak
        offpeak_indicator = QLabel()
        offpeak_indicator.setFixedSize(10, 10)
        offpeak_indicator.setStyleSheet("""
            background-color: #5599FF;
            border-radius: 5px;
            border: none;
        """)
        
        offpeak_label = QLabel("Off-Peak Time:")
        offpeak_label.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            font-weight: bold;
            color: #CCCCCC;
            border: none;
        """)
        offpeak_label.setFixedWidth(100)  # Fixed width for label
        
        # Using same accent color as peak time value
        self.offpeak_time_value = QLabel("--:--:--")
        self.offpeak_time_value.setStyleSheet(f"""
            font-family: Arial;
            font-size: 14px;
            font-weight: bold;
            color: {ACCENT_COLOR};
            border: none;
        """)
        self.offpeak_time_value.setFixedWidth(60)  # Fixed width for time value
        
        self.offpeak_count_value = QLabel("(0 people)")
        self.offpeak_count_value.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #AAAAAA;
            border: none;
        """)
        
        offpeak_row_layout.addWidget(offpeak_indicator)
        offpeak_row_layout.addWidget(offpeak_label)
        offpeak_row_layout.addWidget(self.offpeak_time_value)
        offpeak_row_layout.addWidget(self.offpeak_count_value)
        offpeak_row_layout.addStretch(1)  # Push everything to the left
        
        # Add rows to container
        rows_layout.addWidget(peak_row)
        rows_layout.addWidget(offpeak_row)
        
        # Add container to main layout
        peak_layout.addWidget(rows_container)
        
        return peak_time_widget

    def setup_dropdown_style(self, combobox):
        """Apply consistent styling to dropdown menus"""
        combobox.setStyleSheet(f"""
            QComboBox {{
                background-color: {DROPDOWN_BG_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                padding: 6px 12px;
                {DEFAULT_FONT}
                font-size: 14px;
                min-height: 20px;
            }}
            
            QComboBox:hover {{
                border: 1px solid {ACCENT_COLOR};
            }}
            
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 20px;
                border-left: none;
                padding-right: 5px;
            }}
            
            QComboBox QAbstractItemView {{
                background-color: {PANEL_BG_COLOR};
                border: 1px solid {BORDER_COLOR};
                selection-background-color: {ACCENT_COLOR};
            }}
            
            QComboBox QAbstractItemView::item {{
                padding: 6px 12px;
                min-height: 20px;
            }}
            
            QComboBox QAbstractItemView::item:hover {{
                background-color: {BORDER_COLOR};
            }}
        """)
    
    def setup_main_content(self):
        # Create main content area with video and stats side by side
        main_content = QWidget()
        main_content_layout = QVBoxLayout(main_content)
        main_content_layout.setContentsMargins(0, 0, 0, 0)
        main_content_layout.setSpacing(16)
        
        # Create a splitter for resizable sections
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)  # Prevent sections from being collapsed
        self.main_splitter.setHandleWidth(4)  # Width of the splitter handle
        self.main_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: transparent; /* Make handle invisible */
                margin: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {ACCENT_COLOR};
            }}
        """)
        self.main_splitter.setStretchFactor(0, 1)  # Video container can stretch
        self.main_splitter.setStretchFactor(1, 0)  # Metrics container has fixed width preference
        self.main_splitter.splitterMoved.connect(self.on_splitter_moved)
        
        # Create containers for video output and metrics panel
        video_container = QWidget()
        metrics_container = QWidget()
        
        # Set a minimum width for the metrics container to prevent UI elements from being hidden
        metrics_container.setMinimumWidth(350)  # Set minimum width to ensure all elements are visible
        
        # Setup video output (left side)
        self.setup_video_output(video_container)
        
        # Setup metrics panel (right side)
        self.setup_metrics_panel(metrics_container)
        
        # Add containers to splitter
        self.main_splitter.addWidget(video_container)
        self.main_splitter.addWidget(metrics_container)
        
        # Set initial sizes (70% video, 30% metrics)
        self.main_splitter.setSizes([700, 300])
        
        # Add splitter to main content layout
        main_content_layout.addWidget(self.main_splitter)
        
        # Add main content to main layout with stretch factor
        self.main_layout.addWidget(main_content, 1)
    
    def on_splitter_moved(self, pos, index):
        """Handle splitter movement to update video frame"""
        # Force a resize event to update the video display
        if self.displayed_frame is not None and self.paused:
            rgb_frame = cv2.cvtColor(self.displayed_frame, cv2.COLOR_BGR2RGB)
            self.display_frame(rgb_frame)

    def setup_video_output(self, parent_widget):
        # Apply styling directly to the parent widget
        parent_widget.setStyleSheet(f"""
            background-color: {PANEL_BG_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 6px;
        """)
        
        # Create layout on the parent widget
        output_layout = QVBoxLayout(parent_widget)
        output_layout.setContentsMargins(16, 16, 16, 16)
        output_layout.setSpacing(12)
        
        # Header and controls container - fix: remove any borders here
        header_container = QWidget()
        header_container.setStyleSheet("border: none;")
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)
        
        # Video Feed header WITHOUT border
        output_header = QLabel("Video Feed with Detection:")
        output_header.setStyleSheet(HEADER_FONT_STYLE)
        
        # Add toggle switch for heatmap
        heatmap_container = QWidget()
        heatmap_container.setStyleSheet("border: none;")
        heatmap_layout = QHBoxLayout(heatmap_container)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)
        heatmap_layout.setSpacing(8)
        
        heatmap_label = QLabel("Show Density Heatmap:")
        heatmap_label.setStyleSheet(SUBHEADER_FONT_STYLE)
        
        self.heatmap_toggle = ToggleSwitch()
        self.heatmap_toggle.toggled.connect(self.on_heatmap_toggled)
        self.heatmap_toggle.setEnabled(False)  # Initially disabled since no video is playing
        
        heatmap_layout.addWidget(heatmap_label)
        heatmap_layout.addWidget(self.heatmap_toggle)
        
        # Add output header and toggle to header container
        header_layout.addWidget(output_header)
        header_layout.addStretch(1)
        header_layout.addWidget(heatmap_container)
        
        # Create timer display
        timer_container = QWidget()
        timer_container.setStyleSheet("border: none;")
        timer_layout = QHBoxLayout(timer_container)
        timer_layout.setContentsMargins(0, 0, 0, 0)
        timer_layout.setSpacing(8)
        
        timer_label = QLabel("Elapsed Time:")
        timer_label.setStyleSheet(SUBHEADER_FONT_STYLE)
        
        self.timer_display = QLabel("00:00:00:000")
        self.timer_display.setStyleSheet(f"""
            {DEFAULT_FONT}
            font-size: 14px;
            font-weight: bold;
            color: {ACCENT_COLOR};
            border: none;
            min-width: 80px;
        """)
        
        # Create a container for timer and end indicator
        timer_display_container = QWidget()
        timer_display_container.setStyleSheet("border: none;")
        timer_display_layout = QHBoxLayout(timer_display_container)
        timer_display_layout.setContentsMargins(0, 0, 0, 0)
        timer_display_layout.setSpacing(4)

        # Add existing timer display to this container
        timer_display_layout.addWidget(self.timer_display)

        # Create end of playback indicator
        self.end_playback_label = QLabel("(End of playback reached)")
        self.end_playback_label.setStyleSheet("color: #999999; font-size: 12px;") # Changed to grey, removed italic
        self.end_playback_label.setVisible(False)  # Initially hidden

        timer_display_layout.addWidget(self.end_playback_label)
        timer_display_layout.addStretch(1)
        
        timer_layout.addWidget(timer_label)

        timer_layout.addWidget(timer_display_container)
        timer_layout.addStretch(1)
        
        # Model loading indicator
        self.model_status = QLabel("YOLO Model: Not Loaded")
        self.model_status.setStyleSheet(f"""
            {DEFAULT_FONT}
            font-size: 12px;
            color: {MUTED_TEXT_COLOR};
            border: none;
        """)
        
        # Model loading progress bar
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 100)
        self.model_progress.setValue(0)
        self.model_progress.setVisible(False)
        self.model_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                background-color: {PANEL_BG_COLOR};
                height: 6px;
                text-align: center;
                color: {TEXT_COLOR};
            }}
            
            QProgressBar::chunk {{
                background-color: {ACCENT_COLOR};
            }}
        """)
        
        # Video container
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        video_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create custom video label with drag and drop
        self.video_label = DragDropVideoLabel()
        self.video_label.set_parent_app(self)
        self.video_label.set_default_content()
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumWidth(100)  # Set a very small minimum width
        
        video_layout.addWidget(self.video_label)
        
        # Add components to output layout
        output_layout.addWidget(header_container)
        output_layout.addWidget(timer_container)
        output_layout.addWidget(self.model_status)
        output_layout.addWidget(self.model_progress)
        output_layout.addWidget(video_container, 1)  # Give video container stretch priority

        # Add spacing before export container
        output_layout.addSpacing(8)  # Fixed spacing of 8px

        # Create Export Heatmap button container
        export_container = QWidget()
        export_container.setStyleSheet("border: none; margin: 0;")  # Remove border
        export_layout = QHBoxLayout(export_container)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(10)

        # Heatmap export button
        self.export_heatmap_button = QPushButton("Export Heatmap")
        self.export_heatmap_button.setStyleSheet(EXPORT_BUTTON_STYLE)
        self.export_heatmap_button.setFixedWidth(150)
        self.export_heatmap_button.setEnabled(False)  # Initially disabled
        self.export_heatmap_button.clicked.connect(self.export_heatmap)

        # Add button to layout with left alignment
        export_layout.addWidget(self.export_heatmap_button)
        export_layout.addStretch(1)  # This pushes the button to the left

        # Add button to output layout
        output_layout.addWidget(export_container)
    
    def setup_metrics_panel(self, parent_widget):
        """Setup the metrics panel with improved vertical space handling"""
        # Apply styling directly to parent widget - REMOVED BORDER
        parent_widget.setStyleSheet(f"""
            background-color: {PANEL_BG_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 6px;
        """)
        
        # Create a main layout for the parent widget
        main_layout = QVBoxLayout(parent_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create a scroll area to handle limited vertical space
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: {PANEL_BG_COLOR};
                width: 8px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_COLOR};
                min-height: 20px;
                border-radius: 4px 4px 0 0;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {ACCENT_COLOR};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: none;
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)
        
        # Create a container widget for the scroll area content
        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"border: 1px solid {BORDER_COLOR}; border-bottom: none;")
        metrics_layout = QVBoxLayout(scroll_content)
        metrics_layout.setContentsMargins(16, 16, 16, 16)
        metrics_layout.setSpacing(16)  # Slightly reduced spacing
        
        # Metrics header
        metrics_header = QLabel("Detection Metrics:")
        metrics_header.setStyleSheet("font-family: Arial; font-size: 16px; font-weight: bold; color: #E0E0E0; border: none;")
        
        # Create widgets
        people_count_widget = self.create_people_count_widget()
        crowd_detection_widget = self.create_crowd_detection_widget()
        people_graph_widget = self.create_people_graph_widget()
        peak_time_widget = self.create_peak_time_widget()
        
        # Add the widgets to the layout
        metrics_layout.addWidget(metrics_header)
        metrics_layout.addWidget(people_count_widget)
        metrics_layout.addWidget(crowd_detection_widget)
        metrics_layout.addWidget(people_graph_widget, 1)  # Graph should stretch
        metrics_layout.addWidget(peak_time_widget)
        
        # Set the content widget for the scroll area
        scroll_area.setWidget(scroll_content)
        
        # Create a container for the export button that stays at the bottom
        export_container = QWidget()
        export_container.setStyleSheet("background-color: transparent; border: none;")
        export_layout = QVBoxLayout(export_container)
        export_layout.setContentsMargins(16, 8, 16, 16)
        export_layout.setSpacing(0)
        
        # People Count Graph export button
        self.export_graph_button = QPushButton("Export People Count Graph")
        self.export_graph_button.setStyleSheet(EXPORT_BUTTON_STYLE)
        self.export_graph_button.setFixedWidth(220)
        self.export_graph_button.setEnabled(False)
        self.export_graph_button.clicked.connect(self.export_count_graph)
        
        # Button container for left alignment
        button_container = QWidget()
        button_container.setStyleSheet("background-color: transparent; border: none;")
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)
        button_layout.addWidget(self.export_graph_button)
        button_layout.addStretch(1)
        
        # Add button to export container
        export_layout.addWidget(button_container)
        
        # Add both containers to the main layout
        main_layout.addWidget(scroll_area, 1)  # Scroll area should stretch
        main_layout.addWidget(export_container)  # Export container stays at bottom
        
    def create_people_count_widget(self):
        """Create more compact people count widget with no border"""
        people_count_widget = QWidget()
        people_count_widget.setStyleSheet(f"""
            background-color: {WIDGET_BG_COLOR};
            border-radius: 4px;
        """)
        
        # Use horizontal layout instead of vertical to save space
        people_count_layout = QHBoxLayout(people_count_widget)
        people_count_layout.setContentsMargins(12, 10, 12, 10)  # Reduced padding
        people_count_layout.setSpacing(10)
        
        people_count_header = QLabel("People Detected")
        people_count_header.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #CCCCCC;
            border: none;
        """)
        
        self.people_count_value = QLabel("0")
        self.people_count_value.setStyleSheet(f"""
            font-family: Arial;
            font-size: 32px;
            font-weight: bold;
            color: {ACCENT_COLOR};
            border: none;
        """)
        self.people_count_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.people_count_value.setMinimumWidth(60)  # Increased width for larger numbers
        
        people_count_layout.addWidget(people_count_header)
        people_count_layout.addStretch(1)
        people_count_layout.addWidget(self.people_count_value)
        
        # Set a fixed height to keep it compact but accommodate larger text
        people_count_widget.setFixedHeight(60)  # Increased from 50 to 60
        
        return people_count_widget
    
    def create_people_graph_widget(self):
        # Setup people count graph
        self.setup_people_count_graph()
        
        # Create container widget
        people_graph_widget = QWidget()
        people_graph_widget.setStyleSheet(f"""
            background-color: {WIDGET_BG_COLOR};
            border-radius: 4px;
        """)
        
        # Set the container to also stretch vertically
        people_graph_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding  # Allow vertical expansion
        )
        
        people_graph_layout = QVBoxLayout(people_graph_widget)
        people_graph_layout.setContentsMargins(16, 16, 16, 16)
        people_graph_layout.setSpacing(8)
        
        people_graph_header = QLabel("People Count Over Time")
        people_graph_header.setStyleSheet(f"""
            {DEFAULT_FONT}
            font-size: 14px;
            color: #CCCCCC;
            border: none;
        """)
        
        people_graph_layout.addWidget(people_graph_header)
        people_graph_layout.addSpacing(8)
        people_graph_layout.addWidget(self.people_graph_widget, 1)  # Add stretch factor of 1
        
        return people_graph_widget
    
    def setup_people_count_graph(self):
        """Setup the real-time people count graph with a modern look"""
        # Data for the graph - no maxlen to keep all data
        self.people_data = []
        self.time_data = []
        self.start_time = time.time()
        
        # Create a pyqtgraph PlotWidget
        self.people_graph_widget = pg.PlotWidget()
        
        # Set size policy to expand and shrink with available space
        self.people_graph_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Expanding
        )
        
        # Set minimum height (smaller than before)
        self.people_graph_widget.setMinimumHeight(150)
        
        # Set background to match app theme
        self.people_graph_widget.setBackground(WIDGET_BG_COLOR)
        
        # Remove ALL borders
        self.people_graph_widget.getPlotItem().setContentsMargins(0, 0, 0, 0)
        self.people_graph_widget.setContentsMargins(0, 0, 0, 0)
        self.people_graph_widget.getPlotItem().getViewBox().setBorder(pen=None)
        self.people_graph_widget.setFrameShape(QFrame.Shape.NoFrame)
        
        # Create the plot line with improved styling
        self.people_graph = self.people_graph_widget.plot(
            [], [], 
            pen=pg.mkPen(color=ACCENT_COLOR, width=3), 
            symbolBrush=pg.mkBrush(LIGHTER_ACCENT_COLOR),
            symbolPen=pg.mkPen(LIGHTER_ACCENT_COLOR),
            symbolSize=4,
            symbol='o'
        )
        
        # Configure axes with modern styling
        axis_color = '#888888'
        self.people_graph_widget.getPlotItem().getAxis('bottom').setPen(pg.mkPen(color=axis_color, width=1))
        self.people_graph_widget.getPlotItem().getAxis('left').setPen(pg.mkPen(color=axis_color, width=1))
        
        # Style the axis labels
        self.people_graph_widget.setLabel('left', 'People Count', color='#CCCCCC')
        self.people_graph_widget.setLabel('bottom', 'Time (s)', color='#CCCCCC')
        
        # Style the grid with more subtle lines
        self.people_graph_widget.showGrid(x=True, y=True, alpha=0.2)

    def create_crowd_detection_widget(self):
        """Create widget for crowd threshold detection and alerts"""
        # Main widget container
        crowd_widget = QWidget()
        crowd_widget.setStyleSheet(f"""
            background-color: {WIDGET_BG_COLOR};
            border-radius: 4px;
        """)
        
        # Main layout for the widget
        crowd_layout = QVBoxLayout(crowd_widget)
        crowd_layout.setContentsMargins(12, 12, 12, 12)
        crowd_layout.setSpacing(8)
        
        # Header with toggle switch (always visible)
        header_container = QWidget()
        header_container.setStyleSheet("background-color: transparent; border: none;")
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        
        crowd_header = QLabel("Crowd Threshold Alerts")
        crowd_header.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #CCCCCC;
            border: none;
        """)
        
        self.crowd_toggle = ToggleSwitch()
        self.crowd_toggle.toggled.connect(self.on_crowd_detection_toggled)
        self.crowd_toggle.setEnabled(False)  # Initially disabled since no video is playing
        
        header_layout.addWidget(crowd_header)
        header_layout.addStretch(1)
        header_layout.addWidget(self.crowd_toggle)
        
        # Container for settings (will be shown/hidden)
        self.crowd_settings_container = QWidget()
        self.crowd_settings_container.setStyleSheet("background-color: transparent; border: none;")
        settings_layout = QVBoxLayout(self.crowd_settings_container)
        settings_layout.setContentsMargins(0, 8, 0, 0)  # Add top margin for spacing
        settings_layout.setSpacing(12)  # Slightly increased spacing
        
        # Initially hide the settings
        self.crowd_settings_container.setVisible(False)
        
        # People threshold slider - set integer_display=True
        threshold_container = QWidget()
        threshold_container.setStyleSheet("background-color: transparent; border: none;")
        threshold_layout = QHBoxLayout(threshold_container)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(8)
        
        threshold_label = QLabel("People Threshold:")
        threshold_label.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #E0E0E0;
            border: none;
        """)
        
        self.people_threshold_slider = ModernBoxedSlider(integer_display=True)  # Set to integer display
        self.people_threshold_slider.setRange(3, 50)  # Adjustable range for people count
        self.people_threshold_slider.setValue(self.crowd_size_threshold)
        self.people_threshold_slider.valueChanged.connect(self.on_crowd_size_threshold_changed)
        
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.people_threshold_slider, 1)
        
        # Smoothing window slider - set integer_display=True
        smoothing_container = QWidget()
        smoothing_container.setStyleSheet("background-color: transparent; border: none;")
        smoothing_layout = QHBoxLayout(smoothing_container)
        smoothing_layout.setContentsMargins(0, 0, 0, 0)
        smoothing_layout.setSpacing(8)
        
        smoothing_label = QLabel("Smoothing Window:")
        smoothing_label.setStyleSheet("""
            font-family: Arial;
            font-size: 14px;
            color: #E0E0E0;
            border: none;
        """)
        
        self.smoothing_slider = ModernBoxedSlider(integer_display=True)  # Set to integer display
        self.smoothing_slider.setRange(1, 60)  # 1-60 frames
        self.smoothing_slider.setValue(self.smoothing_window_size)  # Default from init
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_window_changed)
        
        smoothing_layout.addWidget(smoothing_label)
        smoothing_layout.addWidget(self.smoothing_slider, 1)
        
        # Alert status indicator (keeping background color but removing border)
        self.alert_container = QWidget()
        self.alert_container.setStyleSheet(f"""
            background-color: #2A2A2A;
            border-radius: 4px;
            border: 1px solid {BORDER_COLOR};
        """)
        
        alert_layout = QHBoxLayout(self.alert_container)
        alert_layout.setContentsMargins(10, 8, 10, 8)
        alert_layout.setSpacing(8)
        
        self.alert_icon = QLabel("🔔")
        self.alert_icon.setStyleSheet("""
            font-family: Arial;
            font-size: 16px;
            color: #555555;
            border: none;
        """)
        
        self.alert_text = QLabel("People count is normal")
        self.alert_text.setStyleSheet("""
            font-family: Arial;
            font-size: 12px;
            color: #AAAAAA;
            border: none;
        """)
        
        alert_layout.addWidget(self.alert_icon)
        alert_layout.addWidget(self.alert_text, 1)
        
        # Add components to the settings container
        settings_layout.addWidget(threshold_container)
        settings_layout.addWidget(smoothing_container)
        settings_layout.addWidget(self.alert_container)
        
        # Add header and settings container to main layout
        crowd_layout.addWidget(header_container)
        crowd_layout.addWidget(self.crowd_settings_container)
        
        return crowd_widget

    def on_crowd_detection_toggled(self, enabled):
        """Handle crowd detection toggle switch changes"""
        self.crowd_detection_enabled = enabled
        
        # Make sure toggle visual state matches functionality
        self.crowd_toggle.setChecked(enabled)
        
        # Simply show/hide the settings container
        self.crowd_settings_container.setVisible(enabled)
        
        # Reset alert state when turned off
        if not enabled:
            self.update_crowd_alert_status(False)
            self.threshold_alert_active = False

    def on_crowd_size_threshold_changed(self, value):
        """Handle crowd size threshold slider change"""
        self.crowd_size_threshold = value
        
        # Reset alert status when threshold is changed
        self.update_crowd_alert_status(False)

    def on_smoothing_window_changed(self, value):
        """Handle smoothing window size slider change"""
        # Store the current history values
        current_history = list(self.people_count_history)
        
        # Update the window size
        self.smoothing_window_size = value
        
        # Create a new deque with the new size
        self.people_count_history = deque(maxlen=value)
        
        # Add back the existing history, limited to the new size
        for count in current_history[-value:]:
            self.people_count_history.append(count)
            
        # Recalculate smoothed count
        if len(self.people_count_history) > 0:
            self.smoothed_people_count = round(np.mean(self.people_count_history))
            self.people_count_value.setText(str(self.smoothed_people_count))
            
        # Check threshold with new window size
        if self.crowd_detection_enabled and self.current_frame is not None:
            self.check_threshold_crossing(self.current_frame)

    def update_crowd_alert_status(self, alert_active, count=0):
        """Update the crowd alert status indicator"""
        self.threshold_alert_active = alert_active
        
        if alert_active:
            # Active alert styling - light red border
            self.alert_container.setStyleSheet("""
                background-color: #4e1c1c;
                border-radius: 4px;
                border: 1px solid #cc3232; /* Light red border */
            """)
            self.alert_icon.setStyleSheet("""
                font-family: Arial;
                font-size: 16px;
                color: #ff5555;
                border: none;
            """)
            self.alert_text.setStyleSheet("""
                font-family: Arial;
                font-size: 12px;
                color: #ff9999;
                border: none;
            """)
            
            # Update alert text
            self.alert_text.setText(f"ALERT! {count} people detected (threshold: {self.crowd_size_threshold})")
            
            # Also update people count display with alert styling
            if hasattr(self, 'people_count_value'):
                self.people_count_value.setStyleSheet("color: #ff5555; font-size: 32px; font-weight: bold; border: none;")
            
            # Add to alert history with timestamp
            current_time = self.format_time_for_filename(self.video_time_ms)
            alert_record = {
                'timestamp': current_time,
                'count': count,
                'threshold': self.crowd_size_threshold
            }
            self.threshold_history.append(alert_record)
            
        else:
            # Normal status styling - grey border
            self.alert_container.setStyleSheet(f"""
                background-color: #2A2A2A;
                border-radius: 4px;
                border: 1px solid {BORDER_COLOR}; /* Grey border */
            """)
            self.alert_icon.setStyleSheet("""
                font-family: Arial;
                font-size: 16px;
                color: #555555;
                border: none;
            """)
            self.alert_text.setStyleSheet("""
                font-family: Arial;
                font-size: 12px;
                color: #AAAAAA;
                border: none;
            """)
            
            # Reset alert text
            self.alert_text.setText("People count is normal")
            
            # Reset people count display styling
            if hasattr(self, 'people_count_value'):
                self.people_count_value.setStyleSheet(f"color: {ACCENT_COLOR}; font-size: 32px; font-weight: bold; border: none;")
    
    def restart_video(self):
        """Restart the current video from the beginning in a thread-safe way"""
        if self.cap is None or not self.cap.isOpened():
            return
            
        # Store if video was at the end (should auto-play if true)
        video_was_at_end = self.end_playback_label.isVisible()
        
        # Store current pause state
        was_paused = self.paused
        
        # Hide end of playback label
        self.end_playback_label.setVisible(False)
        
        # Always pause video thread first to prevent concurrent access
        self.paused = True
        self.video_thread.pause(True)
        
        # Stop the video thread completely to ensure we have exclusive access to cap
        video_was_running = self.video_thread.running
        if video_was_running:
            self.video_thread.stop()
        
        # Create a small delay to ensure thread has stopped
        QApplication.processEvents()
        time.sleep(0.1)  # Small delay to ensure thread cleanup
        
        # Reset video position to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Reset timer
        self.video_time_ms = 0
        self.last_frame_time = time.time()
        self.update_timer_display()
        
        # Reset graph data
        self.people_data.clear()
        self.time_data.clear()
        
        # Initialize new heatmap accumulator if needed but keep heatmap enabled state
        heatmap_was_enabled = self.heatmap_enabled
        if heatmap_was_enabled:
            self.heatmap_accumulator = None
            
        # Reset threshold alert state
        self.threshold_alert_active = False
        self.update_crowd_alert_status(False)
        if hasattr(self, 'threshold_line') and self.threshold_line is not None:
            self.threshold_line.setData([0, 60], [self.crowd_size_threshold, self.crowd_size_threshold])
        if hasattr(self, 'alert_segment') and self.alert_segment is not None:
            self.people_graph_widget.removeItem(self.alert_segment)
            self.alert_segment = None
        
        # Reset peak tracking
        self.peak_count = 0
        self.peak_time_ms = 0
        self.offpeak_count = float('inf')
        self.offpeak_time_ms = 0
        self.peak_time_value.setText("--:--:--")
        self.peak_count_value.setText("(0 people)")
        self.offpeak_time_value.setText("--:--:--")
        self.offpeak_count_value.setText("(0 people)")
        
        if self.peak_marker is not None:
            self.people_graph_widget.removeItem(self.peak_marker)
            self.peak_marker = None
        
        if self.offpeak_marker is not None:
            self.people_graph_widget.removeItem(self.offpeak_marker)
            self.offpeak_marker = None
        
        # Manually read first frame (thread-safe now that video thread is stopped)
        ret, first_frame = self.cap.read()
        if ret:
            # Process the first frame directly
            self.current_frame = first_frame.copy()
            
            # Apply YOLO detection if ready
            if self.yolo_ready:
                # Run YOLO detection directly to get boxes
                results = self.yolo_thread.model(first_frame, classes=0)
                
                # Collect people boxes for heatmap
                boxes = []
                for result in results:
                    result_boxes = result.boxes
                    for box in result_boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        confidence = float(box.conf[0])
                        
                        # Only count if confidence is above threshold
                        if confidence > self.confidence_threshold:
                            boxes.append((x1, y1, x2, y2))
                
                # Store these boxes
                self.last_detected_boxes = boxes
                
                # Apply heatmap if enabled
                if heatmap_was_enabled:
                    display_frame = self.process_frame_with_heatmap(first_frame, boxes)
                else:
                    display_frame = first_frame.copy()
                    
                # Convert to RGB for display
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Store the processed frame
                self.displayed_frame = display_frame
            else:
                # If YOLO is not ready, just display the frame without processing
                rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                self.displayed_frame = first_frame.copy()
            
            # Display the frame
            self.display_frame(rgb_frame)
            
            # Reset to beginning again since we read one frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Restart video thread if it was running before
        if video_was_running:
            self.video_thread.set_capture(self.cap)
            self.video_thread.start()
        
        # Determine if we should be paused or playing
        # If video was at the end, always start playing regardless of previous state
        if video_was_at_end:
            was_paused = False  # Override pause state to force play
            
        # Restore pause state
        self.paused = was_paused
        self.video_thread.pause(was_paused)
        
        # Update button states
        if was_paused:
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
        else:
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)

    def populate_sources(self):
        # Try to find sources directory
        sources_dir = os.path.join(os.getcwd(), "sources")
        
        if os.path.exists(sources_dir) and os.path.isdir(sources_dir):
            # List all files in the sources directory
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            for file in os.listdir(sources_dir):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    self.source_combo.addItem(file, os.path.join(sources_dir, file))
            
            # If no videos found, add a placeholder
            if self.source_combo.count() == 0:
                self.source_combo.addItem("No video files found in 'sources' directory", "")
        else:
            # If sources directory doesn't exist, add a placeholder
            self.source_combo.addItem("'sources' directory not found", "")
    
    def process_frame_with_heatmap(self, frame, boxes):
        """Process a frame with or without heatmap overlay"""
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Apply heatmap overlay if enabled
        if self.heatmap_enabled:
            # Update heatmap with new positions - this adds to the accumulator
            heatmap = self.update_heatmap(display_frame, boxes)
            
            if heatmap is not None and np.max(heatmap) > 0:
                # Ensure minimum value of 0.1 for blue background in low activity areas
                viz_heatmap = np.maximum(heatmap, 0.1)
                
                # Convert to 8-bit for colormap
                viz_heatmap_8bit = (viz_heatmap * 255).astype(np.uint8)
                
                # Apply JET colormap to get blue->green->red gradient
                heatmap_colored = cv2.applyColorMap(viz_heatmap_8bit, cv2.COLORMAP_JET)
                
                # Darken the original frame to make heatmap more visible
                darkened_frame = cv2.addWeighted(display_frame, 0.4, np.zeros_like(display_frame), 0.6, 0)
                
                # Blend the heatmap with the darkened original frame
                display_frame = cv2.addWeighted(heatmap_colored, 0.7, darkened_frame, 0.3, 0)
                
                # Add grid lines for better visualization
                h, w = display_frame.shape[:2]
                grid_spacing = 50
                
                # Draw vertical grid lines
                for x in range(0, w, grid_spacing):
                    cv2.line(display_frame, (x, 0), (x, h), GRID_COLOR, 1)
                
                # Draw horizontal grid lines
                for y in range(0, h, grid_spacing):
                    cv2.line(display_frame, (0, y), (w, y), GRID_COLOR, 1)
        
        return display_frame

    def update_heatmap(self, frame, boxes):
        """Update the heatmap accumulator with new people positions using a low-resolution approach"""
        h, w = frame.shape[:2]
        
        # Use the class property for scale factor
        scale_factor = self.heatmap_scale_factor
        low_h, low_w = int(h * scale_factor), int(w * scale_factor)
        
        # Check if the frame resolution has changed
        if self.heatmap_accumulator is not None:
            current_low_h, current_low_w = self.heatmap_accumulator.shape
            if current_low_h != low_h or current_low_w != low_w:
                self.heatmap_accumulator = None
                self.aggregate_heatmap_accumulator = None
                self.aggregate_frame_count = 0
        
        # Initialize low-resolution heatmap accumulator if not exists
        if self.heatmap_accumulator is None:
            self.heatmap_accumulator = np.zeros((low_h, low_w), dtype=np.float32)
        
        # Initialize aggregate heatmap accumulator if not exists
        if self.aggregate_heatmap_accumulator is None:
            self.aggregate_heatmap_accumulator = np.zeros((low_h, low_w), dtype=np.float32)
        
        # Apply decay to existing heatmap (only the regular one, not the aggregate)
        self.heatmap_accumulator *= self.heatmap_decay
        
        # Create a new low-resolution heatmap for current positions
        current_heatmap = np.zeros((low_h, low_w), dtype=np.float32)
        
        # Add detected people positions to current heatmap
        for box in boxes:
            # Scale down coordinates to low resolution
            x1, y1, x2, y2 = box
            
            # Use bottom center of bounding box (feet position)
            foot_x = int((x1 + x2) / 2 * scale_factor)
            foot_y = int(y2 * scale_factor)
            
            # Ensure coordinates are within frame
            if 0 <= foot_x < low_w and 0 <= foot_y < low_h:
                # Add a point with intensity 1.0
                current_heatmap[foot_y, foot_x] = 1.0
                
                # Also add some intensity to surrounding pixels for better blending
                # Use the class property for neighbor radius
                radius = self.heatmap_neighbor_radius
                y_min = max(0, foot_y - radius)
                y_max = min(low_h - 1, foot_y + radius)
                x_min = max(0, foot_x - radius)
                x_max = min(low_w - 1, foot_x + radius)
                
                for y in range(y_min, y_max + 1):
                    for x in range(x_min, x_max + 1):
                        # Skip the center point (already set to 1.0)
                        if x == foot_x and y == foot_y:
                            continue
                        # Add lower intensity to neighbors
                        dist = np.sqrt((x - foot_x)**2 + (y - foot_y)**2)
                        if dist <= radius:
                            intensity = 1.0 - (dist / radius) * 0.7
                            current_heatmap[y, x] = max(current_heatmap[y, x], intensity)
        
        # Apply multiple blur passes with different kernel sizes if there are detections
        if np.sum(current_heatmap) > 0:
            # First pass: Small kernel to smooth initial points
            current_heatmap = cv2.GaussianBlur(current_heatmap, (7, 7), 0)
            
            # Second pass: Medium kernel for wider spread
            current_heatmap = cv2.GaussianBlur(current_heatmap, (17, 17), 0)
            
            # Third pass: Large kernel for final smoothing
            current_heatmap = cv2.GaussianBlur(current_heatmap, (31, 31), 0)
            
            # Normalize the current heatmap
            max_val = np.max(current_heatmap)
            if max_val > 0:
                current_heatmap = current_heatmap / max_val
        
        # Add current heatmap to accumulator with appropriate intensity
        self.heatmap_accumulator += current_heatmap * self.heatmap_intensity
        
        # Add to aggregate heatmap accumulator without decay
        if np.sum(current_heatmap) > 0:
            self.aggregate_heatmap_accumulator += current_heatmap
            self.aggregate_frame_count += 1
        
        # Cap the maximum value to prevent overflow
        max_val = np.max(self.heatmap_accumulator)
        if max_val > 1.0:
            self.heatmap_accumulator = self.heatmap_accumulator / max_val
        
        # Upsample back to original resolution for display
        return cv2.resize(self.heatmap_accumulator, (w, h), interpolation=cv2.INTER_LINEAR)
        
    def update_people_graph(self, count):
        """Update the people count graph with new data and threshold line"""
        # Only update when playing video
        if self.cap is None or not self.cap.isOpened() or self.paused:
            return
        
        # Use video time in seconds for x-axis
        current_time_sec = self.video_time_ms / 1000.0
        
        # Add current time and count to data
        self.time_data.append(current_time_sec)
        self.people_data.append(count)
        
        # Update the graph
        self.people_graph.setData(self.time_data, self.people_data)
        
        # Add or update threshold line if crowd detection is enabled
        if self.crowd_detection_enabled:
            # Check if threshold line exists already
            if not hasattr(self, 'threshold_line') or self.threshold_line is None:
                # Create a new threshold line - dashed horizontal line
                pen = pg.mkPen(color='r', width=1, style=Qt.PenStyle.DashLine)
                self.threshold_line = self.people_graph_widget.plot(
                    [0, max(current_time_sec + 10, 60)], 
                    [self.crowd_size_threshold, self.crowd_size_threshold],
                    pen=pen,
                    name='Threshold'
                )
            else:
                # Update existing threshold line
                self.threshold_line.setData(
                    [0, max(current_time_sec + 10, 60)], 
                    [self.crowd_size_threshold, self.crowd_size_threshold]
                )
                
            # Color the graph line to show alert regions
            if self.threshold_alert_active:
                # Last point is above threshold, use red for the last segment
                red_pen = pg.mkPen(color='r', width=3)
                if len(self.time_data) >= 2:
                    # Create or update the alert segment line
                    if not hasattr(self, 'alert_segment') or self.alert_segment is None:
                        self.alert_segment = self.people_graph_widget.plot(
                            self.time_data[-2:], self.people_data[-2:],
                            pen=red_pen
                        )
                    else:
                        self.alert_segment.setData(
                            self.time_data[-2:], self.people_data[-2:]
                        )
        elif hasattr(self, 'threshold_line') and self.threshold_line is not None:
            # Remove threshold line if crowd detection is disabled
            self.people_graph_widget.removeItem(self.threshold_line)
            self.threshold_line = None
            
            # Remove alert segment if it exists
            if hasattr(self, 'alert_segment') and self.alert_segment is not None:
                self.people_graph_widget.removeItem(self.alert_segment)
                self.alert_segment = None
        
        # Always show the full time range from 0 to current time
        padding = max(current_time_sec * 0.05, 1.0)  # At least 1s padding
        if current_time_sec > 0:
            self.people_graph_widget.setXRange(0, current_time_sec + padding)
        
        # Adjust y-axis range with some padding
        if len(self.people_data) > 0:
            max_count = max(max(self.people_data), self.crowd_size_threshold if self.crowd_detection_enabled else 1)
            min_count = min(min(self.people_data), 0)  # Ensure not negative
            y_padding = max((max_count - min_count) * 0.1, 1)  # At least 1 count padding
            self.people_graph_widget.setYRange(
                max(0, min_count - y_padding),  # Don't go below 0
                max_count + y_padding
            )

    def on_threshold_changed(self, value):
        """Handle confidence threshold slider change"""
        # Convert slider value (10-90) to threshold (0.1-0.9)
        self.confidence_threshold = value / 100.0
        
        # Update YOLO thread with new threshold
        self.yolo_thread.set_confidence_threshold(self.confidence_threshold)
    
    def on_heatmap_toggled(self, enabled):
        """Handle heatmap toggle switch changes"""
        self.heatmap_enabled = enabled
        
        # Make sure toggle visual state matches functionality
        self.heatmap_toggle.setChecked(enabled)
        
        # Enable/disable export button based on heatmap state
        self.export_heatmap_button.setEnabled(enabled)
        
        # If video is paused, reprocess the current frame
        if self.paused and self.current_frame is not None and len(self.last_detected_boxes) > 0:
            # Process the current frame with the new heatmap setting
            display_frame = self.process_frame_with_heatmap(self.current_frame, self.last_detected_boxes)
            
            # Store the updated displayed frame
            self.displayed_frame = display_frame.copy()
            
            # Convert to RGB and display
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            self.display_frame(rgb_frame)
    
    def on_model_changed(self, index):
        """Handle model selection change"""
        if index < 0:
            return
            
        # Get selected model key
        model_key = self.model_combo.itemData(index)
        if not model_key or model_key == self.current_model_key:
            return
            
        # Update current model
        self.current_model_key = model_key
        model_info = self.available_models[model_key]
        
        # Check if model exists
        model_path = os.path.join(self.models_dir, model_info["path"])
        
        # If model doesn't exist in models_dir, check the current directory
        if not os.path.exists(model_path):
            # Check if it exists in the current directory
            if os.path.exists(model_info["path"]):
                # It exists in the current directory, use that
                model_path = model_info["path"]
            else:
                # Neither in models_dir nor in current directory, need to download
                self.download_model(model_key)
                return
        
        # Model exists, update path and reload
        self.model_path = model_path
        self.yolo_ready = False
        
        # Update status and start loading
        self.model_status.setText(f"YOLO Model: Loading {model_key}...")
        self.model_progress.setRange(0, 0)  # Indeterminate progress
        self.model_progress.setVisible(True)
        
        # Set new model path and restart YOLO thread if it's running
        if self.yolo_thread.running:
            self.yolo_thread.stop()
            
        self.yolo_thread.set_model_path(self.model_path)
        self.yolo_thread.start()  # This will trigger loading the new model
    
    def download_model(self, model_key):
        """Download the selected model"""
        if self.model_downloading:
            return
            
        model_info = self.available_models[model_key]
        model_path = os.path.join(self.models_dir, model_info["path"])
        
        # Update status
        self.model_status.setText(f"YOLO Model: Downloading {model_key}...")
        self.model_progress.setValue(0)
        self.model_progress.setRange(0, 100)
        self.model_progress.setVisible(True)
        self.model_downloading = True
        
        # Create and start download thread
        self.download_thread = ModelDownloadThread(
            model_key, 
            model_info["url"], 
            model_path
        )
        self.download_thread.progress_update.connect(self.on_download_progress)
        self.download_thread.download_complete.connect(self.on_download_complete)
        self.download_thread.start()
    
    def on_download_progress(self, percentage, message):
        """Update download progress in UI"""
        self.model_status.setText(message)
        self.model_progress.setValue(percentage)
    
    def on_download_complete(self, success, model_path):
        """Handle model download completion"""
        self.model_downloading = False
        
        if success:
            # Update model path
            self.model_path = model_path
            
            # Don't automatically load the model right after download
            # Just update the UI to show the download is complete
            self.model_status.setText(f"YOLO Model: Downloaded successfully")
            self.model_progress.setVisible(False)
            
            # Set flag to indicate the model needs to be loaded
            self.yolo_ready = False
            
        else:
            # Download failed
            self.model_status.setText("YOLO Model: Download failed!")
            self.model_progress.setVisible(False)
            
            # Reset to default model
            default_index = list(self.available_models.keys()).index("YOLOv8n (Nano)")
            self.model_combo.setCurrentIndex(default_index)
    
    def on_model_loaded(self, success, message):
        """Handle model loading completion"""
        if success:
            self.yolo_ready = True
            self.model_status.setText(f"YOLO Model: {self.current_model_key} loaded successfully")
            self.model_progress.setVisible(False)
        else:
            self.yolo_ready = False
            self.model_status.setText(f"YOLO Model: Loading failed - {message}")
            self.model_progress.setVisible(False)
    
    def open_file_dialog(self):
        """Open file dialog for selecting video files"""
        file_filter = "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", file_filter
        )
        
        if file_path:
            self.load_video_from_path(file_path)
    
    def load_video_from_path(self, file_path):
        """Load and play video from the given file path"""
        # Stop any existing video playback
        if self.video_thread.running:
            self.video_thread.stop()
        
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        
        if not os.path.exists(file_path):
            self.video_label.set_default_content()
            return
        
        # Reset video timer
        self.video_time_ms = 0
        self.last_frame_time = 0
        self.update_timer_display()
        
        # Reset graph data
        self.people_data.clear()
        self.time_data.clear()
        
        # Make sure model is loaded
        if not self.yolo_ready:
            if self.yolo_thread.running:
                # Model is already loading, show message and continue
                self.model_status.setText("YOLO Model: Loading... (video will play when ready)")
            else:
                # Start loading model
                self.yolo_thread.set_model_path(self.model_path)
                self.yolo_thread.start()
                self.model_status.setText("YOLO Model: Loading... (video will play when ready)")
                self.model_progress.setRange(0, 0)  # Indeterminate progress
                self.model_progress.setVisible(True)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.video_label.set_default_content()
            return

        self.heatmap_toggle.setEnabled(True)  # Enable heatmap toggle when video is loaded
        self.crowd_toggle.setEnabled(True)    # Enable crowd detection toggle when video is loaded
        self.export_graph_button.setEnabled(True)  # Enable graph export when video loaded

        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.frame_interval = int(1000 / fps)  # ms between frames
        else:
            self.frame_interval = 33  # Default to ~30 fps
        
        # Configure and start the video thread
        self.video_thread.set_capture(self.cap)
        self.paused = False
        self.video_thread.pause(False)
        self.video_thread.start()
        
        # Update button states
        self.play_button.setEnabled(False)  # Disable play button during playback
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)

        self.restart_button.setEnabled(True)
    
    def start_video(self):
        if self.cap is not None and self.cap.isOpened() and self.paused:
            # Resume paused video
            self.paused = False
            self.video_thread.pause(False)
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            return

        self.end_playback_label.setVisible(False)

        # Start new video from dropdown selection
        if self.cap is not None and self.cap.isOpened():
            self.stop_video()
        
        # Get selected video path
        selected_index = self.source_combo.currentIndex()
        video_path = self.source_combo.itemData(selected_index)
        
        if not video_path or not os.path.exists(video_path):
            self.video_label.set_default_content()
            return
        
        # Reset graph data when starting a new video
        self.people_data.clear()
        self.time_data.clear()
        
        # Reset video timer
        self.video_time_ms = 0
        self.last_frame_time = 0
        self.update_timer_display()
        
        self.restart_button.setEnabled(True)  # Enable restart button when playing

        self.load_video_from_path(video_path)
    
    def pause_video(self):
        if self.cap is not None and self.cap.isOpened():
            if not self.paused:
                # Pause video
                self.paused = True
                self.video_thread.pause(True)
                self.play_button.setEnabled(True)
                self.pause_button.setEnabled(False)
            else:
                # Resume video
                self.paused = False
                self.video_thread.pause(False)
                self.play_button.setEnabled(False)
                self.pause_button.setEnabled(True)
    
    def stop_video(self):
        """Stop video playback and reset all visualizations with improved thread handling"""
        # First, pause everything to prevent thread conflicts
        was_playing = not self.paused
        self.paused = True
        if self.video_thread is not None:
            self.video_thread.pause(True)
        
        # Stop video thread completely and ensure it's actually stopped
        if self.video_thread is not None and self.video_thread.running:
            self.video_thread.stop()
            
            # Process events to let thread cleanup happen
            QApplication.processEvents()
            # Small delay to ensure clean thread termination
            time.sleep(0.2)
        
        # Pause YOLO processing and clear its queue
        if hasattr(self, 'yolo_thread') and self.yolo_thread is not None:
            self.yolo_thread.frame_queue = []
            self.yolo_thread.processing = False
        
        # Release video capture with proper exception handling
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception as e:
                print(f"Error releasing video capture: {e}")
            finally:
                self.cap = None
        
        # Make sure the video thread no longer has a reference to the capture
        if self.video_thread is not None:
            self.video_thread.cap = None
        
        # Reset video state
        self.paused = False
        self.current_frame = None
        self.displayed_frame = None
        
        # Force UI update
        QApplication.processEvents()
        
        # Reset the video label to default state - this should update the UI
        self.video_label.set_default_content()
        
        # Reset people count and smoothing history
        self.people_count = 0
        self.people_count_history.clear()
        self.smoothed_people_count = 0
        self.people_count_value.setText("0")
        
        # Reset video timer
        self.video_time_ms = 0
        self.last_frame_time = 0
        self.update_timer_display()
        
        # Reset heatmap accumulator
        self.heatmap_accumulator = None
        self.aggregate_heatmap_accumulator = None
        self.aggregate_frame_count = 0
        
        # Completely clear the graph widget and recreate the plot
        self.people_graph_widget.clear()
        
        # Recreate the plot line for future use
        self.people_graph = self.people_graph_widget.plot(
            [], [], 
            pen=pg.mkPen(color=ACCENT_COLOR, width=3), 
            symbolBrush=pg.mkBrush(LIGHTER_ACCENT_COLOR),
            symbolPen=pg.mkPen(LIGHTER_ACCENT_COLOR),
            symbolSize=4,
            symbol='o'
        )
        
        # Reset data arrays
        self.people_data = []
        self.time_data = []
        
        # Reset peak tracking
        self.peak_count = 0
        self.peak_time_ms = 0
        self.offpeak_count = float('inf')
        self.offpeak_time_ms = 0
        self.peak_time_value.setText("--:--:--")
        self.peak_count_value.setText("(0 people)")
        self.offpeak_time_value.setText("--:--:--")
        self.offpeak_count_value.setText("(0 people)")
        
        # No need to remove markers since we cleared the entire graph widget
        self.peak_marker = None
        self.offpeak_marker = None
        self.threshold_line = None
        self.alert_segment = None
        
        # Reset threshold alert state
        self.threshold_alert_active = False
        if self.crowd_detection_enabled:
            self.update_crowd_alert_status(False)
            
        # Hide end of playback label
        self.end_playback_label.setVisible(False)

        # Turn off heatmap if it was on and disable the toggle
        if self.heatmap_enabled:
            self.heatmap_enabled = False
            self.heatmap_toggle.setChecked(False)
        self.heatmap_toggle.setEnabled(False)

        # Also make sure to disable the export buttons
        self.export_heatmap_button.setEnabled(False)
        self.export_graph_button.setEnabled(False)

        # Update button states
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.restart_button.setEnabled(False)
        
        # Force UI to update again
        QApplication.processEvents()
        self.repaint()

    def on_video_ended(self):
        """Handle video reaching the end"""
        # Always pause when video reaches the end
        self.paused = True
        self.video_thread.pause(True)
        
        # Show end of playback indicator
        self.end_playback_label.setVisible(True)
        
        # Update button states - disable both play and pause buttons
        self.play_button.setEnabled(False)  # Disable play button at end of video
        self.pause_button.setEnabled(False)
        self.restart_button.setEnabled(True)  # Enable restart button
    
    def update_timer_display(self):
        """Update the timer display with the current video time"""
        # Calculate hours, minutes, seconds, milliseconds
        total_seconds = self.video_time_ms // 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = self.video_time_ms % 1000
        
        # Format the time string
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
        
        # Update the display
        self.timer_display.setText(time_str)
        
        # Force update of UI
        QApplication.processEvents()
    
    def process_video_frame(self, frame):
        """Process video frame and send to YOLO detection thread"""
        if frame is None:
            return
        
        # Check if the video thread has detected a loop
        if self.video_thread.loop_detected:
            self.video_thread.loop_detected = False  # Reset flag
            
            # Reset timer
            self.video_time_ms = 0
            self.last_frame_time = time.time()
            self.update_timer_display()
            
            # Reset graph data
            self.people_data.clear()
            self.time_data.clear()
            
            # Reset heatmap accumulator if needed
            if self.heatmap_enabled and self.heatmap_accumulator is not None:
                self.heatmap_accumulator = None
        
        # Update video timer (only if not paused)
        if not self.paused:
            current_time = time.time()
            if self.last_frame_time > 0:
                elapsed = int((current_time - self.last_frame_time) * 1000)  # ms
                # Limit elapsed time to reasonable values to avoid jumps during resize
                if elapsed < 1000:  # Cap at 1 second to prevent huge jumps
                    self.video_time_ms += elapsed
            self.last_frame_time = current_time
            self.update_timer_display()
        
        # Store the current frame for resize events
        self.current_frame = frame.copy()
        
        # Convert frame to RGB (from BGR) for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Send the frame to YOLO thread for detection if model is loaded
        if self.yolo_ready:
            self.yolo_thread.add_frame(frame)
        else:
            # If YOLO is not ready, display the frame without detection
            self.display_frame(rgb_frame)
            # Make sure to store the displayed frame
            self.displayed_frame = frame.copy()

    def update_peak_time_display(self):
        """Update peak and off-peak time displays"""
        # Format peak time
        if self.peak_time_ms > 0:
            peak_hours = (self.peak_time_ms // 1000) // 3600
            peak_minutes = ((self.peak_time_ms // 1000) % 3600) // 60
            peak_seconds = (self.peak_time_ms // 1000) % 60
            peak_time_str = f"{peak_hours:02d}:{peak_minutes:02d}:{peak_seconds:02d}"
            self.peak_time_value.setText(peak_time_str)
            self.peak_count_value.setText(f"({self.peak_count} people)")  # Added parentheses
            
            # Update peak marker on graph - removed white border
            if len(self.time_data) > 0:
                peak_time_sec = self.peak_time_ms / 1000.0
                if self.peak_marker is None:
                    # Create marker if it doesn't exist - removed symbolPen parameter
                    self.peak_marker = self.people_graph_widget.plot(
                        [peak_time_sec], [self.peak_count],
                        pen=None, symbol='o', symbolSize=10,
                        symbolBrush='#FF5555'  # Pure red with no border
                    )
                else:
                    # Update existing marker
                    self.peak_marker.setData([peak_time_sec], [self.peak_count])
        
        # Format off-peak time
        if self.offpeak_time_ms > 0 and self.offpeak_count < float('inf'):
            offpeak_hours = (self.offpeak_time_ms // 1000) // 3600
            offpeak_minutes = ((self.offpeak_time_ms // 1000) % 3600) // 60
            offpeak_seconds = (self.offpeak_time_ms // 1000) % 60
            offpeak_time_str = f"{offpeak_hours:02d}:{offpeak_minutes:02d}:{offpeak_seconds:02d}"
            self.offpeak_time_value.setText(offpeak_time_str)
            self.offpeak_count_value.setText(f"({self.offpeak_count} people)")  # Added parentheses
            
            # Update off-peak marker on graph - removed white border
            if len(self.time_data) > 0:
                offpeak_time_sec = self.offpeak_time_ms / 1000.0
                if self.offpeak_marker is None:
                    # Create marker if it doesn't exist - removed symbolPen parameter
                    self.offpeak_marker = self.people_graph_widget.plot(
                        [offpeak_time_sec], [self.offpeak_count],
                        pen=None, symbol='o', symbolSize=10,
                        symbolBrush='#5599FF'  # Pure blue with no border
                    )
                else:
                    # Update existing marker
                    self.offpeak_marker.setData([offpeak_time_sec], [self.offpeak_count])

    def display_detection_results(self, processed_frame, people_count, boxes):
        """Display processed frame with detections, heatmap, and update people count"""
        if processed_frame is None:
            return
        
        # Store the last detected boxes for use when toggling heatmap while paused
        self.last_detected_boxes = boxes.copy()
        
        # Add current count to history for smoothing
        self.people_count_history.append(people_count)
        
        # Calculate smoothed people count (moving average)
        if len(self.people_count_history) > 0:
            self.smoothed_people_count = round(np.mean(self.people_count_history))
        else:
            self.smoothed_people_count = people_count
        
        # Update people count display with smoothed value
        self.people_count = self.smoothed_people_count
        self.people_count_value.setText(str(self.smoothed_people_count))
        
        # Check for threshold crossing if crowd detection is enabled
        if self.crowd_detection_enabled:
            self.check_threshold_crossing(processed_frame)
        
        # Update the people count graph with smoothed value
        self.update_people_graph(self.smoothed_people_count)

        # Track peak and off-peak
        if self.smoothed_people_count > self.peak_count:
            self.peak_count = self.smoothed_people_count
            self.peak_time_ms = self.video_time_ms
            self.update_peak_time_display()
            
        if self.smoothed_people_count < self.offpeak_count and self.smoothed_people_count > 0:
            # Only track non-zero off-peak to avoid counting before people appear
            self.offpeak_count = self.smoothed_people_count
            self.offpeak_time_ms = self.video_time_ms
            self.update_peak_time_display()
        
        # Store the original frame
        self.current_frame = processed_frame.copy()
        
        # Process the frame with or without heatmap
        display_frame = self.process_frame_with_heatmap(processed_frame, boxes)
        
        # Add threshold alert visualization if active
        if self.crowd_detection_enabled and self.threshold_alert_active:
            # Add red border to indicate alert
            h, w = display_frame.shape[:2]
            # Top border
            display_frame[0:8, 0:w] = [0, 0, 200]
            # Bottom border
            display_frame[h-8:h, 0:w] = [0, 0, 200]
            # Left border
            display_frame[0:h, 0:8] = [0, 0, 200]
            # Right border
            display_frame[0:h, w-8:w] = [0, 0, 200]
            
            # Add alert text
            alert_text = f"ALERT! {self.smoothed_people_count} people (threshold: {self.crowd_size_threshold})"
            cv2.putText(display_frame, alert_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Store the final displayed frame (with heatmap if enabled)
        self.displayed_frame = display_frame.copy()
        
        # Convert to RGB for display
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Display the processed frame
        self.display_frame(rgb_frame)

    def check_threshold_crossing(self, frame):
        """Check if people count exceeds threshold using the smoothed value"""
        # Changed from >= to > for stricter comparison
        if self.smoothed_people_count > self.crowd_size_threshold:
            # Count exceeds threshold - activate alert if not already active
            if not self.threshold_alert_active:
                self.update_crowd_alert_status(True, self.smoothed_people_count)
        else:
            # Count is below threshold or equal - deactivate alert if currently active
            if self.threshold_alert_active:
                self.update_crowd_alert_status(False)
    
    def display_frame(self, rgb_frame):
        """Display a video frame"""
        if rgb_frame is None:
            return
        
        # Convert to QImage and then to QPixmap
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Get the current size of the video label
        label_size = self.video_label.size()
        
        # Scale the pixmap to fit the label size while preserving aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(label_size, 
                                    Qt.AspectRatioMode.KeepAspectRatio, 
                                    Qt.TransformationMode.SmoothTransformation)
        
        # Hide the content layout widgets before showing video
        for i in range(self.video_label.content_layout.count()):
            item = self.video_label.content_layout.itemAt(i)
            if item and item.widget():
                item.widget().setVisible(False)
        
        # Display the frame
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def resizeEvent(self, event):
        """Handle window resize events without affecting video playback"""
        super().resizeEvent(event)
        
        # Always update the display when resizing, even when paused
        if self.displayed_frame is not None:
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(self.displayed_frame, cv2.COLOR_BGR2RGB)
            self.display_frame(rgb_frame)
        elif self.current_frame is not None:
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            self.display_frame(rgb_frame)
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop the video thread
        if self.video_thread.running:
            self.video_thread.stop()
        
        # Stop the YOLO thread
        if self.yolo_thread.running:
            self.yolo_thread.stop()
        
        # Release video capture
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        
        event.accept()
    
    def format_time_for_filename(self, time_ms):
        """Format time in milliseconds to a string suitable for filenames"""
        # Calculate hours, minutes, seconds
        total_seconds = time_ms // 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        milliseconds = time_ms % 1000
        
        # Format the time string
        return f"{hours:02d}h{minutes:02d}m{seconds:02d}s{milliseconds:03d}ms"

    def export_count_graph(self):
        """Export the people count graph as an image"""
        # Check if we have graph data
        if len(self.time_data) == 0 or len(self.people_data) == 0:
            self.show_export_error_message("No graph data available yet.")
            return
        
        # Ask the user to select an output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", os.path.join(os.getcwd(), "exports"),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not output_dir:  # User canceled
            return
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a high-resolution figure
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        
        # Create a figure with high DPI for quality output
        fig = Figure(figsize=(10, 6), dpi=150)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Plot the data with styling
        ax.plot(list(self.time_data), list(self.people_data), 
                marker='o', markersize=4, linewidth=2, color=ACCENT_COLOR)
        
        # Style the plot
        ax.set_facecolor('#2D2D30')  # Match app background
        fig.patch.set_facecolor('#252526')  # Match app panel color
        
        # Grid styling
        ax.grid(True, linestyle='--', alpha=0.3, color='#888888')
        
        # Spine styling (borders)
        for spine in ax.spines.values():
            spine.set_color('#3E3E42')
        
        # Set labels and title
        ax.set_xlabel('Time (seconds)', color='#CCCCCC')
        ax.set_ylabel('People Count', color='#CCCCCC')
        ax.set_title('People Count Over Time', color='#FFFFFF', fontsize=14)
        
        # Style the ticks
        ax.tick_params(colors='#CCCCCC')
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the figure
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_dir, f"people_count_graph_{timestamp}.png")
        fig.savefig(output_path)
        
        # Cleanup
        plt.close(fig)
        
        # Show success message
        self.show_export_success_message(output_path)

    def export_heatmap(self):
        """Export the aggregate heatmap directly after selecting a directory"""
        # First, check if we have aggregate heatmap data
        if self.aggregate_heatmap_accumulator is None or self.aggregate_frame_count <= 0:
            self.show_export_error_message("No heatmap data available yet. Play a video with heatmap enabled first.")
            return
        
        # Ask the user to select an output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", os.path.join(os.getcwd(), "exports"),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not output_dir:  # User canceled
            return
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a normalized version of the aggregate heatmap
        normalized_aggregate = self.aggregate_heatmap_accumulator.copy()
        
        # Normalize by the number of frames that contributed to it
        if self.aggregate_frame_count > 0:
            normalized_aggregate /= self.aggregate_frame_count
        
        # Scale up to original frame size
        h, w = self.current_frame.shape[:2] if self.current_frame is not None else (720, 1280)
        heatmap = cv2.resize(normalized_aggregate, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Apply additional blur for smoother visualization
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # Convert to colormap
        heatmap_8bit = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        
        # Create a background frame for the heatmap
        if self.current_frame is not None:
            background = cv2.addWeighted(self.current_frame, 0.4, np.zeros_like(self.current_frame), 0.6, 0)
            result = cv2.addWeighted(heatmap_colored, 0.7, background, 0.3, 0)
        else:
            result = heatmap_colored
        
        # Save the result
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_dir, f"aggregate_heatmap_{timestamp}.png")
        cv2.imwrite(output_path, result)
        
        # Show success message
        self.show_export_success_message(output_path)

    def show_export_success_message(self, output_path):
        """Show success message for heatmap export"""
        from PyQt6.QtWidgets import QMessageBox
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Export Complete")
        msg.setText("Heatmap export completed successfully!")
        
        # If it's a directory, offer to open it
        if os.path.isdir(output_path):
            msg.setInformativeText(f"Heatmap snapshots saved to:\n{output_path}")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Open)
            result = msg.exec()
            
            if result == QMessageBox.StandardButton.Open:
                # Open the directory in file explorer
                import subprocess
                import platform
                
                if platform.system() == "Windows":
                    os.startfile(output_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(["open", output_path])
                else:  # Linux and other Unix-like
                    subprocess.Popen(["xdg-open", output_path])
        else:
            msg.setInformativeText(f"Heatmap saved to:\n{output_path}")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()

    def show_export_error_message(self, error_msg):
        """Show error message for heatmap export"""
        from PyQt6.QtWidgets import QMessageBox
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Export Error")
        msg.setText("Error exporting heatmap")
        msg.setInformativeText(error_msg)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    window = CrowdSenseApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()    