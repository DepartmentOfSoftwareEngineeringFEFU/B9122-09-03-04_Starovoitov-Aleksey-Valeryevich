import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import sys
import json
from datetime import datetime, timedelta
import glob
from collections import defaultdict
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def read_image_unicode(image_path):
    try:
        img = cv2.imread(image_path)
        if img is not None:
            return img
        
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return img
    except Exception:
        return None

HOME = r'C:\dev\Diplom'
EMBEDDINGS_DIR = f'{HOME}/valid_embeddings'

os.makedirs(f'{HOME}/datasets', exist_ok=True)
os.makedirs(f'{HOME}/runs/detect', exist_ok=True)
os.makedirs(f'{HOME}/known_faces', exist_ok=True)
os.makedirs(f'{HOME}/video_input', exist_ok=True)
os.makedirs(f'{HOME}/images_input', exist_ok=True)
os.makedirs(f'{HOME}/results', exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def delete_old_json_files(results_dir, hours=1):
    if not os.path.exists(results_dir):
        return
    
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=hours)
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(results_dir, filename)
            try:
                if datetime.fromtimestamp(os.path.getctime(file_path)) < cutoff_time:
                    os.remove(file_path)
            except Exception:
                pass

def load_embeddings_from_files(embeddings_dir):
    pickle_file = os.path.join(embeddings_dir, "embeddings_full.pkl")
    
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            embeddings = data["embeddings"]
            names = data["names"]
            return np.array(embeddings), names
    
    return None, None

def get_latest_video(video_dir):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, f'*{ext}')))
        video_files.extend(glob.glob(os.path.join(video_dir, f'*{ext.upper()}')))
    
    if not video_files:
        return None
    
    return max(video_files, key=os.path.getctime)

def clear_images_input(images_dir):
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            file_path = os.path.join(images_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception:
                pass

def extract_frames_from_video(video_path, output_dir, frame_interval=30, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    clear_images_input(output_dir)
    video_name = Path(video_path).stem
    
    extracted_frames = []
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps if fps > 0 else frame_count
            frame_filename = f"{video_name}_frame_{saved_count:06d}_{timestamp:.2f}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
            saved_count += 1
            
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return extracted_frames

def extract_embeddings_batch(face_regions, app):
    embeddings = []
    for face in face_regions:
        if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
            embeddings.append(None)
            continue
        
        faces = app.get(face)
        
        if len(faces) == 0:
            embeddings.append(None)
        else:
            face_obj = max(faces, key=lambda x: x.det_score)
            embeddings.append(face_obj.normed_embedding)
    
    return embeddings

def recognize_faces_batch(embeddings, known_embeddings_norm, known_names, threshold=0.4):
    results = []
    
    for embedding in embeddings:
        if embedding is None:
            results.append(("unknown (no face)", 0.0))
            continue
        
        try:
            query_norm = embedding / np.linalg.norm(embedding)
            similarities = np.dot(known_embeddings_norm, query_norm)
            best_idx = np.argmax(similarities)
            best_similarity = float(similarities[best_idx])
            
            if best_similarity >= threshold:
                results.append((known_names[best_idx], best_similarity))
            else:
                results.append(("unknown", best_similarity))
        except Exception:
            results.append(("error", 0.0))
    
    return results

def process_all_images_batch_yolo(image_dir, yolo_model, app, known_embeddings, known_names, threshold=0.3):
    known_embeddings_norm = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(image_dir, f'*{ext.upper()}')))
    
    if not image_files:
        return {}, 0
    
    all_images = []
    valid_image_paths = []
    
    for img_path in image_files:
        img = read_image_unicode(img_path)
        if img is not None:
            all_images.append(img)
            valid_image_paths.append(img_path)
    
    if not all_images:
        return {}, 0
    
    batch_size = 8
    all_detections = []
    
    for i in range(0, len(all_images), batch_size):
        batch_images = all_images[i:i + batch_size]
        batch_paths = valid_image_paths[i:i + batch_size]
        
        batch_results = yolo_model(batch_images, verbose=False)
        
        for result_idx, result in enumerate(batch_results):
            img_path = batch_paths[result_idx]
            img = batch_images[result_idx]
            
            if len(result.boxes) > 0:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    
                    h, w = y2 - y1, x2 - x1
                    padding_x = int(w * 0.6)
                    padding_y = int(h * 0.6)
                    
                    x1_pad = max(0, x1 - padding_x)
                    y1_pad = max(0, y1 - padding_y)
                    x2_pad = min(img.shape[1], x2 + padding_x)
                    y2_pad = min(img.shape[0], y2 + padding_y)
                    
                    face_region = img[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    all_detections.append({
                        'image_path': img_path,
                        'face_region': face_region
                    })
    
    if not all_detections:
        return {}, 0
    
    best_results = defaultdict(lambda: {"similarity": 0.0, "data": None})
    total_recognized = 0
    
    face_batch_size = 16
    
    for i in range(0, len(all_detections), face_batch_size):
        batch_detections = all_detections[i:i + face_batch_size]
        batch_faces = [d['face_region'] for d in batch_detections]
        
        embeddings = extract_embeddings_batch(batch_faces, app)
        recognition_results = recognize_faces_batch(
            embeddings, known_embeddings_norm, known_names, threshold
        )
        
        for j, (name, similarity) in enumerate(recognition_results):
            if name not in ["unknown", "unknown (no face)", "error"]:
                detection = batch_detections[j]
                if similarity > best_results[name]["similarity"]:
                    best_results[name]["similarity"] = float(similarity)
                    best_results[name]["data"] = {
                        "name": name,
                        "similarity": float(similarity),
                        "image_name": os.path.basename(detection['image_path'])
                    }
                    total_recognized += 1
    
    return best_results, total_recognized

def save_final_results_to_json(best_results, video_name):
    current_datetime = datetime.now()
    date_str = current_datetime.strftime("%Y-%m-%d")
    time_str = current_datetime.strftime("%H-%M-%S")
    datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    
    json_filename = f"{HOME}/results/итоговые_результаты_{video_name}_{date_str}_{time_str}.json"
    
    results_array = []
    for person_name, result_data in best_results.items():
        if result_data["data"] is not None:
            face_data = {
                "имя": person_name,
                "значение_сходства": float(result_data["similarity"]),
                "дата_время_распознавания": datetime_str,
                "название_изображения": result_data["data"]["image_name"]
            }
            results_array.append(face_data)
    
    results_array.sort(key=lambda x: x["значение_сходства"], reverse=True)
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_array, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    return json_filename

def main():
    delete_old_json_files(f'{HOME}/results', hours=1)
    
    known_embeddings, known_names = load_embeddings_from_files(EMBEDDINGS_DIR)
    
    if known_embeddings is None or len(known_embeddings) == 0:
        sys.exit(1)
    
    model_path = f'{HOME}/runs/detect/train/weights/best.pt'
    if os.path.exists(model_path):
        yolo_model = YOLO(model_path)
    else:
        yolo_model = YOLO('yolov8n-face.pt')
    
    try:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        sys.exit(1)
    
    video_path = get_latest_video(f'{HOME}/video_input')
    if video_path is None:
        sys.exit(1)
    
    video_name = Path(video_path).stem
    
    extracted_frames = extract_frames_from_video(
        video_path, f'{HOME}/images_input', frame_interval=30, max_frames=None
    )
    
    if not extracted_frames:
        sys.exit(1)
    
    best_results, total_recognized = process_all_images_batch_yolo(
        f'{HOME}/images_input', yolo_model, app, known_embeddings, known_names, threshold=0.3
    )
    
    if best_results:
        save_final_results_to_json(best_results, video_name)

if __name__ == "__main__":
    main()