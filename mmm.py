import os
import re
import numpy as np
import cv2 as cv
import easyocr

# ==================== HÀM PREPROCESSING TÌM BIỂN SỐ ====================

def find_plate_candidates(frame):
    """
    Tìm các vùng ứng viên biển số từ frame video
    Sử dụng logic từ code xử lý ảnh tĩnh (gray -> binary -> denoising -> canny -> morphology -> contours)
    """
    # Chuyển sang ảnh xám
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Chuyển sang ảnh nhị phân với OTSU thresholding
    _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Làm sạch ảnh (khử nhiễu)
    clean_img = cv.fastNlMeansDenoising(binary, h=10)
    
    # Tìm các cạnh
    edges = cv.Canny(clean_img, 20, 150)
    
    # Morphological closing để tìm các hình chữ nhật kín
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    closed_bbox = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    # Tìm các contours
    contours, _ = cv.findContours(closed_bbox.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Lọc các vùng ứng viên cho biển số
    plates = []
    frame_size = frame.shape[0] * frame.shape[1]
    
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w == 0 or h == 0:
            continue
            
        aspect_ratio = w / h  # Tỷ lệ chiều dài/chiều cao
        area_ratio = (w * h) / frame_size  # Tỷ lệ diện tích biển số so với frame
        
        # Lọc theo tiêu chí: tỷ lệ khung hình và diện tích
        if (1.0 < aspect_ratio < 6.0) and (0.005 < area_ratio < 0.5):
            plates.append((x, y, w, h))
    
    return plates

def crop_plate(img, x, y, w, h, pad=5):
    """Cắt vùng biển số với padding"""
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    return img[y1:y2, x1:x2]

def recognize_plate_text(plate_img, reader):
    """
    Nhận diện text từ ảnh biển số
    """
    # Xử lý ảnh biển số
    gray_plate = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
    _, binary_plate = cv.threshold(gray_plate, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # OCR nhận diện text
    ocr_results = reader.readtext(binary_plate, allowlist="0123456789ABCDEFGHKLMNPQRSTUVXY", detail=1)
    
    plate_text = ""
    plate_conf = 0
    for (bbox, text, conf) in ocr_results:
        plate_text += text
        plate_conf = max(plate_conf, conf)
    
    return plate_text, plate_conf

# ==================== XỬ LÝ VIDEO ====================

video_path = "./plate2.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

reader = easyocr.Reader(["en"], gpu=False)

process_every_n_frames = 4
cached_detections = {"plates": []}
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    plates = []
    
    if frame_index % process_every_n_frames == 0:
        # TÌM CÁC VÙNG BIỂN SỐ BẰNG PREPROCESSING (GIỐNG CODE ẢNH TĨNH)
        plate_candidates = find_plate_candidates(frame)
        
        # OCR TRÊN TỪNG VÙNG ỨNG VIÊN
        for (x, y, w, h) in plate_candidates:
            plate_roi = crop_plate(frame, x, y, w, h)
            
            # Nhận diện text
            plate_text, plate_conf = recognize_plate_text(plate_roi, reader)
            
            if plate_text and plate_conf > 0.3:  # Lọc theo confidence
                plates.append({
                    "text": plate_text,
                    "conf": plate_conf,
                    "box": (x, y, w, h)
                })
                print(f"Frame {frame_index}: Tìm thấy biển số '{plate_text}' (Confidence: {plate_conf:.3f})")
        
        cached_detections["plates"] = plates
    else:
        plates = cached_detections["plates"]
    
    # VẼ KẾT QUẢ TRÊN FRAME
    for plate_data in plates:
        x, y, w, h = plate_data["box"]
        plate_text = plate_data["text"]
        plate_conf = plate_data["conf"]
        
        # Vẽ hình chữ nhật
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Vẽ text và confidence
        label = f"{plate_text} ({plate_conf:.2f})"
        cv.putText(frame, label, (x, max(y - 10, 20)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Hiển thị số frame
    cv.putText(frame, f"Frame: {frame_index}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv.imshow("License Plate Detection - Video", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_index += 1

cap.release()
cv.destroyAllWindows()
print("Xong!")