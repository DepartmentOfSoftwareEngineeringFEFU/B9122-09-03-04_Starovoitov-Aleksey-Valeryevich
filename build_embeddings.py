import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
import json
import pickle
from datetime import datetime
import sys

if sys.platform == 'win32':
    import locale
    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')

HOME = r'C:\dev\Diplom'
KNOWN_FACES_DIR = os.path.join(HOME, 'known_faces')
EMBEDDINGS_DIR = os.path.join(HOME, 'valid_embeddings')

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

try:
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
except Exception:
    try:
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(320, 320))
    except Exception:
        sys.exit(1)

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

def extract_face_embedding(img, app):
    faces = app.get(img)
    
    if len(faces) == 0:
        img_large = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        faces = app.get(img_large)
        
        if len(faces) == 0:
            return None, None
    
    face = max(faces, key=lambda x: x.det_score)
    embedding = face.normed_embedding
    
    return embedding, face.det_score

def get_all_files_recursive(directory):
    files = []
    
    for entry in os.scandir(directory):
        if entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
                files.append(entry.path)
        elif entry.is_dir():
            files.extend(get_all_files_recursive(entry.path))
    
    return files

def build_embeddings_database(known_faces_dir, embeddings_dir):
    if not os.path.exists(known_faces_dir):
        return None
    
    embeddings_data = {
        "embeddings": [],
        "names": [],
        "image_paths": [],
        "det_scores": [],
        "metadata": {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_persons": 0,
            "total_faces": 0,
            "persons": {}
        }
    }
    
    persons_folders = []
    for entry in os.scandir(known_faces_dir):
        if entry.is_dir():
            persons_folders.append(entry)
    
    for person_entry in persons_folders:
        person_name = person_entry.name
        person_folder = person_entry.path
        
        image_files = get_all_files_recursive(person_folder)
        person_face_count = 0
        
        for img_path in image_files:
            img = read_image_unicode(img_path)
            
            if img is None:
                continue
            
            embedding, det_score = extract_face_embedding(img, app)
            
            if embedding is not None:
                embeddings_data["embeddings"].append(embedding)
                embeddings_data["names"].append(person_name)
                embeddings_data["image_paths"].append(img_path)
                embeddings_data["det_scores"].append(float(det_score))
                person_face_count += 1
        
        embeddings_data["metadata"]["persons"][person_name] = {
            "folder": person_folder,
            "total_images": len(image_files),
            "successful_faces": person_face_count
        }
    
    embeddings_data["metadata"]["total_persons"] = len(persons_folders)
    embeddings_data["metadata"]["total_faces"] = len(embeddings_data["embeddings"])
    
    if len(embeddings_data["embeddings"]) > 0:
        embeddings_array = np.array(embeddings_data["embeddings"])
        np.save(os.path.join(embeddings_dir, "embeddings.npy"), embeddings_array)
    else:
        np.save(os.path.join(embeddings_dir, "embeddings.npy"), np.array([]))
    
    with open(os.path.join(embeddings_dir, "names.json"), 'w', encoding='utf-8') as f:
        json.dump(embeddings_data["names"], f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(embeddings_dir, "image_paths.json"), 'w', encoding='utf-8') as f:
        json.dump(embeddings_data["image_paths"], f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(embeddings_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(embeddings_data["metadata"], f, ensure_ascii=False, indent=2)
    
    if len(embeddings_data["embeddings"]) > 0:
        embeddings_array = np.array(embeddings_data["embeddings"])
    else:
        embeddings_array = np.array([])
    
    with open(os.path.join(embeddings_dir, "embeddings_full.pkl"), 'wb') as f:
        pickle.dump({
            "embeddings": embeddings_array,
            "names": embeddings_data["names"],
            "image_paths": embeddings_data["image_paths"],
            "det_scores": embeddings_data["det_scores"],
            "metadata": embeddings_data["metadata"]
        }, f)
    
    return embeddings_data

def main():
    if not os.path.exists(KNOWN_FACES_DIR):
        return
    
    embeddings_data = build_embeddings_database(KNOWN_FACES_DIR, EMBEDDINGS_DIR)

if __name__ == "__main__":
    main()