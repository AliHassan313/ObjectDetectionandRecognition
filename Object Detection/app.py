import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, 
                            QProgressBar, QComboBox, QSlider, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np
from ultralytics import YOLO
import time

class DetectionThread(QThread):
    """Thread to perform object detection without freezing the UI"""
    detection_complete = pyqtSignal(np.ndarray, list)
    progress_update = pyqtSignal(int)
    
    def __init__(self, image, model, conf_threshold):
        super().__init__()
        self.image = image
        self.model = model
        self.conf_threshold = conf_threshold
        
    def run(self):
        # Show we're starting
        self.progress_update.emit(10)
        
        # Perform detection
        results = self.model(self.image, conf=self.conf_threshold)
        self.progress_update.emit(80)
        
        # Process results to get bounding boxes
        result = results[0]
        detections = []
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            detections.append((x1, y1, x2, y2, conf, class_name))
        
        self.progress_update.emit(100)
        self.detection_complete.emit(self.image, detections)


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()
        
    def initUI(self):
        # Set window properties
        self.setWindowTitle('Object Detection System')
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #cccccc;")
        self.image_label.setMinimumSize(800, 600)
        
        # Control panel - top row
        control_panel_top = QHBoxLayout()
        
        # Load Image button
        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.loadImage)
        self.load_button.setStyleSheet("font-size: 14px; padding: 8px 16px;")
        control_panel_top.addWidget(self.load_button)
        
        # Detect Objects button
        self.detect_button = QPushButton('Detect Objects')
        self.detect_button.clicked.connect(self.detectObjects)
        self.detect_button.setStyleSheet("font-size: 14px; padding: 8px 16px;")
        self.detect_button.setEnabled(False)
        control_panel_top.addWidget(self.detect_button)
        
        # Save Result button
        self.save_button = QPushButton('Save Result')
        self.save_button.clicked.connect(self.saveResult)
        self.save_button.setStyleSheet("font-size: 14px; padding: 8px 16px;")
        self.save_button.setEnabled(False)
        control_panel_top.addWidget(self.save_button)
        
        # Control panel - bottom row
        control_panel_bottom = QHBoxLayout()
        
        # Confidence threshold slider
        threshold_layout = QVBoxLayout()
        threshold_label = QLabel('Confidence Threshold:')
        threshold_layout.addWidget(threshold_label)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)  # Default 0.5
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        
        self.threshold_value_label = QLabel(f"Value: {self.threshold_slider.value()/100:.2f}")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_value_label.setText(f"Value: {v/100:.2f}")
        )
        
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        control_panel_bottom.addLayout(threshold_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        control_panel_bottom.addWidget(self.progress_bar)
        
        # Add control panels to main layout
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(control_panel_top)
        main_layout.addLayout(control_panel_bottom)
        
        # Status bar for additional information
        self.statusBar().showMessage('Ready')
        
        # Set the layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Variables for image handling
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.detection_results = None
        
    def loadModel(self):
        """Load the YOLOv8 model"""
        try:
            # Check if model exists in the app directory
            if not os.path.exists('yolov8m.pt'):
                self.statusBar().showMessage('Model file not found. Please place yolov8m.pt in the application directory.')
                QMessageBox.warning(self, "Model Not Found", 
                                    "The YOLOv8m model file (yolov8m.pt) was not found. Please place it in the application directory.")
                return
            
            self.statusBar().showMessage('Loading model...')
            self.model = YOLO('yolov8m.pt')
            self.statusBar().showMessage('Model loaded successfully')
        except Exception as e:
            self.statusBar().showMessage(f'Error loading model: {str(e)}')
            QMessageBox.critical(self, "Model Loading Error", f"Failed to load the YOLOv8 model: {str(e)}")
    
    def loadImage(self):
        """Open file dialog to load an image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.webp)'
        )
        
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is None:
                self.statusBar().showMessage('Failed to load image')
                return
            
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.displayImage(image_rgb)
            
            self.detect_button.setEnabled(True)
            self.save_button.setEnabled(False)
            self.statusBar().showMessage(f'Image loaded: {os.path.basename(file_path)}')
    
    def displayImage(self, image, detections=None):
        """Display image in the label with optional detection boxes"""
        h, w, ch = image.shape
        bytes_per_line = ch * w
        
        # Convert to QImage and then to QPixmap
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # If there are detections, draw them
        if detections:
            painter = QPainter(pixmap)
            for x1, y1, x2, y2, conf, class_name in detections:
                # Assign different colors based on class
                color = self.getColorForClass(class_name)
                painter.setPen(QPen(color, 3))
                
                # Draw rectangle
                painter.drawRect(x1, y1, x2-x1, y2-y1)
                
                # Draw label background
                text = f"{class_name} {conf:.2f}"
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(text)
                text_height = font_metrics.height()
                
                # Draw semi-transparent background for text
                painter.fillRect(
                    x1, y1-text_height-5, text_width+10, text_height+5, 
                    QColor(0, 0, 0, 180)
                )
                
                # Draw text
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(x1+5, y1-5, text)
            
            painter.end()
        
        # Scale the image to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        self.processed_image = image.copy() if detections else None
        
    def getColorForClass(self, class_name):
        """Return a color based on class name"""
        # Dictionary mapping class categories to colors
        color_map = {
            'person': QColor(255, 0, 0),      # Red for people
            'bicycle': QColor(0, 255, 0),     # Green for bicycles
            'car': QColor(0, 0, 255),         # Blue for cars
            'motorcycle': QColor(255, 255, 0), # Yellow for motorcycles
            'truck': QColor(255, 0, 255),     # Magenta for trucks
            'bus': QColor(0, 255, 255),       # Cyan for buses
            'dog': QColor(255, 128, 0),       # Orange for dogs
            'cat': QColor(128, 0, 255),       # Purple for cats
        }
        
        # Common categories
        if class_name.lower() in color_map:
            return color_map[class_name.lower()]
        
        # For other classes, create a consistent but different color
        # Hash the class name to get a consistent color
        hash_val = sum(ord(c) for c in class_name)
        r = (hash_val * 123) % 256
        g = (hash_val * 456) % 256
        b = (hash_val * 789) % 256
        
        return QColor(r, g, b)
    
    def detectObjects(self):
        """Detect objects in the loaded image"""
        if self.original_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        # Get the confidence threshold
        conf_threshold = self.threshold_slider.value() / 100
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        
        # Create and start detection thread
        self.detection_thread = DetectionThread(
            self.original_image, self.model, conf_threshold
        )
        self.detection_thread.detection_complete.connect(self.handleDetectionResults)
        self.detection_thread.progress_update.connect(self.progress_bar.setValue)
        
        # Disable UI elements during detection
        self.detect_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.statusBar().showMessage('Detecting objects...')
        
        # Start detection
        self.detection_thread.start()
    
    def handleDetectionResults(self, image, detections):
        """Handle detection results from the thread"""
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store detection results for later use
        self.detection_results = detections
        
        # Display image with detections
        self.displayImage(image_rgb, detections)
        
        # Re-enable UI elements
        self.detect_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # Count objects by class
        class_counts = {}
        for _, _, _, _, _, class_name in detections:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Format and display results
        result_text = f"Detected {len(detections)} objects: "
        result_text += ", ".join([f"{count} {cls}" for cls, count in class_counts.items()])
        self.statusBar().showMessage(result_text)
    
    def saveResult(self):
        """Save the processed image with detection boxes"""
        if self.processed_image is None or not self.detection_results:
            QMessageBox.warning(self, "No Results", "No detection results to save.")
            return
        
        # Get save file path
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Result', '', 'Image Files (*.png *.jpg *.jpeg)'
        )
        
        if save_path:
            # Get the current pixmap and save it
            pixmap = self.image_label.pixmap()
            if pixmap:
                pixmap.save(save_path)
                self.statusBar().showMessage(f'Result saved as {os.path.basename(save_path)}')
            else:
                self.statusBar().showMessage('Failed to save result')

def main():
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()