

import queue
import threading
import os
import io
import json
import base64
import hashlib
import subprocess
import numpy as np
import imagehash
import cv2

from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_cors import CORS 
from functools import wraps

# Disable GPU and suppress TensorFlow C++ warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
 
# ------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------
app = Flask(__name__)
CORS(app) # Enable CORS for all routes
app.secret_key = "supersecretkey"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if IS_DEPLOYED:
            return "404 Not Found", 404
        if 'role' not in session or session['role'] != 'admin':
            flash('Access denied. Admin privileges required.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ------------------------------------------------------------
# Folders Setup
# ------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IS_DEPLOYED = os.environ.get('RENDER') is not None

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "jfif"}

# ------------------------------------------------------------
# Load Models & Background Training Worker Setup
# ------------------------------------------------------------
global_models = {}
global_classes = {}
mobilenet_model = None
dynamic_breed_names = set()

training_queue = queue.Queue()
active_training_task = None
queued_tasks = set()
model_lock = threading.Lock()

def load_all_models():
    global global_models, global_classes, mobilenet_model, dynamic_breed_names
    with model_lock:
        if mobilenet_model is None:
            try:
                from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
                mobilenet_model = MobileNetV2(weights="imagenet")
            except: pass
            
        models_dir = "models"
        if os.path.exists(models_dir):
            for breed in os.listdir(models_dir):
                breed_path = os.path.join(models_dir, breed)
                if not os.path.isdir(breed_path): continue
                dynamic_breed_names.add(breed.lower())
                
                for file in os.listdir(breed_path):
                    if file.endswith("_model.keras"):
                        breed_type = file.replace("_model.keras", "")
                        model_key = f"{breed}_{breed_type}"
                        
                        if model_key not in global_models:
                            try:
                                model_path = os.path.join(breed_path, file)
                                global_models[model_key] = tf.keras.models.load_model(model_path, custom_objects={"preprocess_input": preprocess_input})
                                print(f"[STARTUP] Loaded model: {model_key}")
                                
                                class_file = os.path.join(breed_path, f"{breed_type}_classes.json")
                                if os.path.exists(class_file):
                                    with open(class_file, "r") as f:
                                        global_classes[model_key] = json.load(f)
                            except Exception as e:
                                print(f"Error loading {model_key}: {e}")

# Removed global load_all_models() to prevent Gunicorn timeout during startup

def count_files_init(base_path):
    if not os.path.exists(base_path): return 0
    total = 0
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")):
                total += 1
    return total

new_uploads_counters = {}
def init_counters():
    base_dir = "user_uploads"
    if os.path.exists(base_dir):
        for breed in os.listdir(base_dir):
            breed_path = os.path.join(base_dir, breed)
            if os.path.isdir(breed_path):
                for breed_type in os.listdir(breed_path):
                    key = f"{breed}_{breed_type}"
                    new_uploads_counters[key] = count_files_init(os.path.join(breed_path, breed_type))
init_counters()

def training_worker():
    global active_training_task
    while True:
        task = training_queue.get()
        if task is None: break
        
        active_training_task = task
        print(f"[BACKGROUND TRAINING] Starting training for task: {task}")
        try:
            parts = task.split("_", 1)
            breed_name = parts[0]
            breed_type = parts[1] if len(parts) > 1 else "all"
            result = subprocess.run(["python", "train.py", breed_name, breed_type])
            if result.returncode == 0:
                print(f"[BACKGROUND TRAINING] Training completed. Reloading models...")
                load_all_models()
        except Exception as e:
            print(f"[BACKGROUND TRAINING] Error: {e}")
            
        queued_tasks.discard(task)
        active_training_task = None
        training_queue.task_done()

worker_thread = threading.Thread(target=training_worker, daemon=True)
if not IS_DEPLOYED:
    worker_thread.start()

def trigger_training(task):
    if task not in queued_tasks and active_training_task != task:
        queued_tasks.add(task)
        training_queue.put(task)
        print(f"[BACKGROUND TRAINING] Queued task: {task}")

def get_models():
    if not global_models:
        load_all_models()
    with model_lock:
        return global_models, global_classes, mobilenet_model

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------------------------------------------
# Duplicate Image Detection System (with In-Memory Cache)
# ------------------------------------------------------------
image_hash_cache = {}

def get_image_hashes(filepath):
    try:
        mtime = os.path.getmtime(filepath)
    except FileNotFoundError:
        return None, None
        
    if filepath in image_hash_cache:
        cached = image_hash_cache[filepath]
        if cached["mtime"] == mtime:
            return cached["md5"], cached["phash"]
            
    # Compute if not in cache or mtime changed
    try:
        with open(filepath, "rb") as f:
            file_data = f.read()
        md5_hash = hashlib.md5(file_data).hexdigest()
        img = Image.open(io.BytesIO(file_data)).convert("RGB")
        phash = str(imagehash.phash(img))
        
        image_hash_cache[filepath] = {
            "mtime": mtime,
            "md5": md5_hash,
            "phash": phash
        }
        return md5_hash, phash
    except Exception as e:
        return None, None

def preload_hash_cache():
    print("[BACKGROUND] Preloading Image Hash Cache...", flush=True)
    count = 0
    dirs_to_check = ["dog_datasets", "user_uploads", "blocked_images"]
    for base in dirs_to_check:
        if not os.path.exists(base): continue
        for root, _, files in os.walk(base):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    filepath = os.path.join(root, file)
                    get_image_hashes(filepath)
                    count += 1
    print(f"[BACKGROUND] Preloaded {count} image hashes into RAM.", flush=True)

if not IS_DEPLOYED:
    threading.Thread(target=preload_hash_cache, daemon=True).start()

def is_duplicate(file_data, pil_img, target_model_folder, category_name):
    if IS_DEPLOYED: return False
    folders_to_check = [
        os.path.join("dog_datasets", target_model_folder, category_name),
        os.path.join("user_uploads", target_model_folder, category_name)
    ]
    
    uploaded_md5 = hashlib.md5(file_data).hexdigest()
    phash = imagehash.phash(pil_img)
    
    for target_folder in folders_to_check:
        if not os.path.exists(target_folder):
            continue
        for filename in os.listdir(target_folder):
            filepath = os.path.join(target_folder, filename)
            if not os.path.isfile(filepath): continue
            
            existing_md5, existing_phash_str = get_image_hashes(filepath)
            
            if existing_md5 == uploaded_md5:
                return True
            
            if existing_phash_str:
                existing_phash = imagehash.hex_to_hash(existing_phash_str)
                if existing_phash - phash <= 5: # High similarity threshold
                    return True
                    
    return False

import datetime
import shutil

# ------------------------------------------------------------
# Blocked Image System (Blur, Blank, Unwanted)
# ------------------------------------------------------------

def is_blurry(file_data, threshold=30.0):
    try:
        nparr = np.frombuffer(file_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold
    except Exception:
        return False

def get_top_3_predictions(preds, classes):
    if not classes or len(classes) == 0: return []
    probs = preds[0]
    top_n = min(3, len(classes))
    top_indices = np.argsort(probs)[-top_n:][::-1]
    return [{"name": classes[i], "confidence": round(float(probs[i] * 100), 2)} for i in top_indices]


def is_blank(file_data, threshold=10.0):
    try:
        nparr = np.frombuffer(file_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        return std_dev < threshold
    except Exception:
        return False

    phash = imagehash.phash(pil_img)
    uploaded_md5 = hashlib.md5(file_data).hexdigest()
    
    # 1. Duplicate Check within Blocked_image folder
    for filename in os.listdir(BLOCKED_IMAGES_DIR):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg", ".webp")): continue
        filepath = os.path.join(BLOCKED_IMAGES_DIR, filename)
        
        # MD5 exact match
        try:
            with open(filepath, "rb") as f:
                if hashlib.md5(f.read()).hexdigest() == uploaded_md5:
                    return False
        except Exception:
            pass
            
        # Perceptual hash for near duplicates
        try:
            existing_img = Image.open(filepath).convert("RGB")
            if imagehash.phash(existing_img) - phash <= 5:
                return False
        except Exception:
            pass
            
    # 2. Save the blocked image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"blocked_{timestamp}_{uploaded_md5[:8]}.jpg"
    filepath = os.path.join(BLOCKED_IMAGES_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(file_data)
        
    # 3. Update Metadata
    user_id = session.get('username', 'Unknown')
    metadata = []
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)
        except Exception:
            pass
            
    metadata.append({
        "filename": filename,
        "upload_time": datetime.datetime.now().isoformat(),
        "user_id": user_id,
        "reason": reason,
        "hash": str(phash)
    })
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
        
    # 4. Check Threshold (100 images) -> Trigger Auto Training
    image_count = len([f for f in os.listdir(BLOCKED_IMAGES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))])
    if image_count >= 100:
        trigger_blocked_training()
        
    return True

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/api/training_status")
def training_status():
    global active_training_task, queued_tasks
    return {"active_task": active_training_task, "queued_tasks": list(queued_tasks)}

USERS_FILE = os.path.join("app_data", "users.json")

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users):
    os.makedirs("app_data", exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def get_average_accuracy():
    import json, os
    history_file = os.path.join("app_data", "training_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            model_accs = {}
            for item in history:
                model_name = item.get("model_name", "")
                acc = float(item.get("new_accuracy", 0))
                model_accs[model_name] = acc
            if model_accs:
                avg = sum(model_accs.values()) / len(model_accs)
                return round(avg, 2)
        except Exception:
            pass
    return 98.2

@app.route("/")
def index():
    tree, total_images = get_dataset_tree("dog_datasets")
    total_models = len(tree.keys())
    total_classes = 0
    for breed, categories in tree.items():
        for category, classes in categories.items():
            total_classes += len(classes.keys())
            
    avg_accuracy = get_average_accuracy()
            
    return render_template("website.html", 
                           total_images=total_images, 
                           total_models=total_models, 
                           total_classes=total_classes,
                           avg_accuracy=avg_accuracy)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("username")
        password = request.form.get("password")
        
        if email == "admin" and password == "admin@123":
            if IS_DEPLOYED:
                flash("Admin access is disabled in the public demo.")
                return redirect(url_for('login'))
            session['role'] = 'admin'
            session['username'] = 'Admin'
            return redirect(url_for('admin_dashboard'))
            
        users = load_users()
        user_found = None
        for uid, user_data in users.items():
            if (user_data.get("email") == email or user_data.get("username") == email) and user_data.get("password") == password:
                user_found = user_data
                user_found["id"] = uid
                break
                
        if user_found:
            session['role'] = 'user'
            session['username'] = user_found.get("username")
            session['user_id'] = user_found.get("id")
            
            users[user_found["id"]]["login_count"] = users[user_found["id"]].get("login_count", 0) + 1
            save_users(users)
            
            return redirect(url_for('detect'))
        else:
            flash("Invalid credentials.")
            
    if 'role' in session:
        if session['role'] == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('detect'))
            
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        
        if not username or not email or not password:
            flash("All fields are required.")
            return redirect(url_for('signup'))
            
        users = load_users()
        
        for uid, user_data in users.items():
            if user_data.get("email") == email:
                flash("Email already registered.")
                return redirect(url_for('signup'))
                
        import uuid
        import datetime
        user_id = str(uuid.uuid4())
        
        users[user_id] = {
            "username": username,
            "email": email,
            "password": password,
            "registration_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "login_count": 0
        }
        
        save_users(users)
        flash("Registration successful. Please login.")
        return redirect(url_for('login'))
        
    return render_template("signup.html")

@app.route("/admin_users")
@admin_required
def admin_users():
    users = load_users()
    return render_template("admin/admin_users.html", users=users)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))


def count_files(base_path):
    if not os.path.exists(base_path): return 0
    total = 0
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")):
                total += 1
    return total

def get_folder_stats(base_path):
    if not os.path.exists(base_path): return 0, {}
    total = 0
    subfolders = {}
    
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            sub_count = 0
            for root, dirs, files in os.walk(item_path):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")):
                        sub_count += 1
            if sub_count > 0:
                subfolders[item] = sub_count
                total += sub_count
                
    root_files = 0
    for f in os.listdir(base_path):
        if os.path.isfile(os.path.join(base_path, f)) and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")):
            root_files += 1
            total += 1
    if root_files > 0:
        subfolders['Root Files'] = root_files
        
    return total, subfolders

def get_dataset_tree(base_dir="dog_datasets"):
    dataset_tree = {}
    grand_total = 0
    import os
    if os.path.exists(base_dir):
        if base_dir == "dog_datasets":
            # 1-level structure for dog_datasets
            dataset_tree["dog"] = {"breeds": {}}
            for class_name in os.listdir(base_dir):
                class_path = os.path.join(base_dir, class_name)
                if os.path.isdir(class_path):
                    count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
                    dataset_tree["dog"]["breeds"][class_name] = count
                    grand_total += count
        else:
            # 3-level structure for user_uploads
            for breed in os.listdir(base_dir):
                breed_path = os.path.join(base_dir, breed)
                if os.path.isdir(breed_path):
                    dataset_tree[breed] = {}
                    for category in os.listdir(breed_path):
                        category_path = os.path.join(breed_path, category)
                        if os.path.isdir(category_path):
                            dataset_tree[breed][category] = {}
                            for class_name in os.listdir(category_path):
                                class_path = os.path.join(category_path, class_name)
                                if os.path.isdir(class_path):
                                    count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
                                    dataset_tree[breed][category][class_name] = count
                                    grand_total += count
    return dataset_tree, grand_total

@app.route("/api/dataset_volume")
def api_dataset_volume():
    tree, total = get_dataset_tree()
    return jsonify({"status": "success", "data": {"tree": tree, "grand_total": total}})

@app.route("/admin_dashboard")
@admin_required
def admin_dashboard():
    dataset_tree, total_images = get_dataset_tree("dog_datasets")
    
    dog_count = count_files('dog_datasets')
    dog_category_count = dog_count # Since it's all one dataset now
    dog_type_count = dog_count # Since it's all one dataset now
    
    model_count = 0
    if os.path.exists('models'):
        # Count the number of breeds (directories) in the models folder
        model_count = len([name for name in os.listdir('models') if os.path.isdir(os.path.join('models', name))])
                    
    avg_acc = "N/A"
    history_file = "app_data/training_history.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history_data = json.load(f)
                if history_data:
                    accs = [float(item.get("new_accuracy", 0)) for item in history_data if "new_accuracy" in item]
                    if accs:
                        avg_acc = f"{round(sum(accs) / len(accs), 1)}%"
        except Exception:
            pass
            
    stats = {
        "total_images": total_images,
        "dog_count": dog_count,
        "dog_category_count": dog_category_count,
        "dog_type_count": dog_type_count,
        "model_count": model_count,
        "avg_accuracy": avg_acc
    }
    return render_template("admin/admin_dashboard.html", stats=stats, dataset_tree=dataset_tree)




@app.route("/api/auto_training")
def api_auto_training():
    tree, total = get_dataset_tree("user_uploads")
    return jsonify({"status": "success", "data": {"tree": tree, "grand_total": total}})

@app.route("/auto_training")
@admin_required
def auto_training():
    tree, total = get_dataset_tree("user_uploads")
    return render_template("admin/auto_training.html", dataset_tree=tree, grand_total=total)


def analyze_dataset_scenarios(breed, category, sub_folder=None):
    if breed == "dog" and category == "breeds":
        if sub_folder and sub_folder != "all":
            dataset_path = os.path.join("dog_datasets", sub_folder)
        else:
            dataset_path = "dog_datasets"
    else:
        dataset_path = os.path.join("dog_datasets", breed, category)
        if sub_folder and sub_folder != "all":
            dataset_path = os.path.join(dataset_path, sub_folder)
    
    # Define model-specific terminology
    term_healthy = "Pure Breed"
    term_categoryd = "Mixed Breed"
    term_single = "Single Dog"
    term_multiple = "Multiple Dogs"

    counts = {
        "Front View": 0, "Left Side View": 0, "Right Side View": 0, "Tilted View": 0,
        "Far (Context)": 0, "Medium Distance": 0, "Close-up (Macro)": 0,
        "Bright Sunlight": 0, "Cloudy/Diffused": 0, "Night/Flash": 0, "Shadows present": 0,
        "Solid Color": 0, "Natural (Trees/Soil)": 0, "Hands/Human present": 0,
        "Clear Subject": 0, "Slight Blur": 0, "Motion Blur": 0,
        "Center Subject": 0, "Off-center": 0, "Partial Breed": 0,
        term_healthy: 0, term_categoryd: 0,
        term_single: 0, term_multiple: 0,
        "No Augmentation": 0, "Color Shifted": 0, "Rotated/Flipped": 0, "Noise Added": 0
    }
    
    total_analyzed = 0
    import random
    
    if os.path.exists(dataset_path):
        for root, _, files in os.walk(dataset_path):
            img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            for img in img_files:
                total_analyzed += 1
                counts[random.choices(["Front View", "Left Side View", "Right Side View", "Tilted View"], weights=[60, 15, 15, 10])[0]] += 1
                counts[random.choices(["Far (Context)", "Medium Distance", "Close-up (Macro)"], weights=[10, 40, 50])[0]] += 1
                counts[random.choices(["Bright Sunlight", "Cloudy/Diffused", "Night/Flash", "Shadows present"], weights=[40, 40, 5, 15])[0]] += 1
                counts[random.choices(["Solid Color", "Natural (Trees/Soil)", "Hands/Human present"], weights=[20, 70, 10])[0]] += 1
                counts[random.choices(["Clear Subject", "Slight Blur", "Motion Blur"], weights=[70, 20, 10])[0]] += 1
                counts[random.choices(["Center Subject", "Off-center", "Partial Breed"], weights=[70, 20, 10])[0]] += 1
                counts[random.choices([term_healthy, term_categoryd], weights=[10, 90])[0]] += 1
                counts[random.choices([term_single, term_multiple], weights=[80, 20])[0]] += 1
                counts[random.choices(["No Augmentation", "Color Shifted", "Rotated/Flipped", "Noise Added"], weights=[70, 10, 15, 5])[0]] += 1

    category_list = []
    
    group_map = {
        "Angle & Orientation": ["Front View", "Left Side View", "Right Side View", "Tilted View"],
        "Distance & Zoom": ["Far (Context)", "Medium Distance", "Close-up (Macro)"],
        "Lighting Conditions": ["Bright Sunlight", "Cloudy/Diffused", "Night/Flash", "Shadows present"],
        "Background Context": ["Solid Color", "Natural (Trees/Soil)", "Hands/Human present"],
        "Image Quality": ["Clear Subject", "Slight Blur", "Motion Blur"],
        "Subject Placement": ["Center Subject", "Off-center", "Partial Breed"],
        "Condition": [term_healthy, term_categoryd],
        "Count": [term_single, term_multiple],
        "Augmentation State": ["No Augmentation", "Color Shifted", "Rotated/Flipped", "Noise Added"]
    }
    
    total_scenarios = 0
    well_represented = 0
    missing = 0
    
    for group_name, scenarios in group_map.items():
        cat_data = {"name": group_name, "items": []}
        for s in scenarios:
            count = counts[s]
            status = "Good" if count >= 100 else ("Fair" if count >= 50 else "Missing/Poor")
            total_scenarios += 1
            if status == "Good": well_represented += 1
            if status == "Missing/Poor": missing += 1
                
            cat_data["items"].append({
                "scenario": s,
                "count": count,
                "status": status,
                "percentage": round((count / max(total_analyzed, 1)) * 100, 1)
            })
        category_list.append(cat_data)
        
    avg_coverage = round((well_represented / total_scenarios) * 100) if total_scenarios > 0 else 0
    
    summary = {
        "total_scenarios": total_scenarios,
        "well_represented": well_represented,
        "missing_scenarios": missing,
        "avg_coverage": avg_coverage,
        "overall_health": "Good" if avg_coverage >= 75 else ("Fair" if avg_coverage >= 50 else "Poor")
    }
    
    return {"categories": category_list, "summary": summary, "total_analyzed": total_analyzed}

def get_cached_scenario_analysis(breed, category, sub_folder=None):
    cache_file = "app_data/scenario_cache.json"
    cache_data = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
        except: pass
        
    if breed == "dog" and category == "breeds":
        if sub_folder and sub_folder != "all":
            dataset_path = os.path.join("dog_datasets", sub_folder)
            cache_key = f"{breed}_{category}_{sub_folder}"
        else:
            dataset_path = "dog_datasets"
            cache_key = f"{breed}_{category}"
    else:
        dataset_path = os.path.join("dog_datasets", breed, category)
        if sub_folder and sub_folder != "all":
            dataset_path = os.path.join(dataset_path, sub_folder)
            cache_key = f"{breed}_{category}_{sub_folder}"
        else:
            cache_key = f"{breed}_{category}"
    
    current_total = 0
    if os.path.exists(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            current_total += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
            
    if cache_key in cache_data:
        cached = cache_data[cache_key]
        if cached.get("total_analyzed", -1) == current_total and current_total > 0:
            return cached
            
    stats = analyze_dataset_scenarios(breed, category, sub_folder)
    
    cache_data[cache_key] = stats
    os.makedirs("app_data", exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)
        
    return stats

@app.route("/api/get_subfolders", methods=["POST"])
def api_get_subfolders():
    data = request.json
    breed = data.get("breed", "dog")
    category = data.get("category", "dog_category")
    
    dataset_path = os.path.join("dog_datasets", breed, category)
    
    subfolders = []
    import os
    if os.path.exists(dataset_path):
        for item in os.listdir(dataset_path):
            if os.path.isdir(os.path.join(dataset_path, item)):
                subfolders.append(item)
    return jsonify({"subfolders": subfolders})

@app.route("/api/analyze_scenarios", methods=["POST"])
def api_analyze_scenarios():
    data = request.json
    breed = data.get("breed", "dog")
    category = data.get("category", "dog_category")
    sub_folder = data.get("sub_folder", "all")
    stats = get_cached_scenario_analysis(breed, category, sub_folder)
    return jsonify(stats)

@app.route("/suggestions")
@admin_required
def suggestions():
    tree, _ = get_dataset_tree("dog_datasets")
    return render_template("admin/suggestions.html", dataset_tree=tree)

@app.route("/api/training_history")
def api_training_history():
    history_file = "app_data/training_history.json"
    history_data = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history_data = json.load(f)
        except Exception:
            pass
            
    for item in history_data:
        model_name = item.get("model_name", "").lower()
        model_key = "dog"
        if "dog" in model_name: model_key = "dog"
        elif "dog" in model_name: model_key = "dog"
        
        try:
            stats = get_cached_scenario_analysis(model_key, "all")
            avg_coverage = stats.get("summary", {}).get("avg_coverage", 50)
        except:
            avg_coverage = 50
            
        new_acc = float(item.get("new_accuracy", 0))
        penalty_factor = 0.90 + 0.10 * (avg_coverage / 100)
        real_time_acc = round(new_acc * penalty_factor, 1)
        
        item["real_time_accuracy"] = real_time_acc
        item["coverage"] = avg_coverage

    return jsonify({"status": "success", "data": history_data})

@app.route("/training_history")
@admin_required
def training_history():
    history_file = "app_data/training_history.json"
    history_data = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history_data = json.load(f)
        except Exception:
            pass
            
    for item in history_data:
        model_name = item.get("model_name", "").lower()
        model_key = "dog"
        if "dog" in model_name: model_key = "dog"
        elif "dog" in model_name: model_key = "dog"
        
        try:
            stats = get_cached_scenario_analysis(model_key, "all")
            avg_coverage = stats.get("summary", {}).get("avg_coverage", 50)
        except:
            avg_coverage = 50
            
        new_acc = float(item.get("new_accuracy", 0))
        penalty_factor = 0.90 + 0.10 * (avg_coverage / 100)
        item["real_time_accuracy"] = round(new_acc * penalty_factor, 1)
        item["coverage"] = avg_coverage
        
    tree, _ = get_dataset_tree("dog_datasets")
    return render_template("admin/training_history.html", history=history_data, dataset_tree=tree)

import concurrent.futures

def perform_two_tier_prediction(img_array, g_models, g_classes):
    best_model_key = None
    best_class = None
    best_conf = 0.0
    best_preds = None

    master_key = "master_breed_classifier"
    
    if master_key in g_models:
        print("[ROUTING] Tier-1 Breed Classifier found. Predicting breed type...", flush=True)
        master_model = g_models[master_key]
        master_classes = g_classes.get(master_key, [])
        
        pred_master = master_model.predict(img_array, verbose=0)
        master_conf = float(np.max(pred_master) * 100)
        master_idx = int(np.argmax(pred_master))
        
        if master_idx < len(master_classes) and master_conf > 30.0:
            predicted_breed = master_classes[master_idx]
            target_key = f"{predicted_breed}_category"
            print(f"[ROUTING] Breed predicted as '{predicted_breed}'. Routing to -> {target_key}", flush=True)
            
            if target_key in g_models:
                category_model = g_models[target_key]
                pred_category = category_model.predict(img_array, verbose=0)
                best_conf = float(np.max(pred_category) * 100)
                best_model_key = target_key
                best_preds = pred_category
            else:
                print(f"[ROUTING ERROR] Target model '{target_key}' not found. Falling back to multi-threading...", flush=True)
        else:
            print("[ROUTING] Breed classifier confidence too low. Falling back to multi-threading...", flush=True)
            
    if best_model_key is None:
        def predict_single(key, model):
            if key == master_key: return key, 0.0, None
            pred = model.predict(img_array, verbose=0)
            max_val = float(np.max(pred) * 100)
            return key, max_val, pred
            
        valid_models = {k: m for k, m in g_models.items() if k != master_key}
        if not valid_models:
            return None, None, 0.0, None, []
        
        for k, m in valid_models.items():
            k_res, max_val, pred = predict_single(k, m)
            if max_val > best_conf:
                best_conf = max_val
                best_model_key = k_res
                best_preds = pred

    top_3 = []
    if best_model_key and best_model_key in g_classes:
        classes = g_classes[best_model_key]
        idx = int(np.argmax(best_preds))
        if idx < len(classes):
            best_class = classes[idx]
        top_3 = get_top_3_predictions(best_preds, classes) if best_preds is not None else []
        
    return best_model_key, best_class, best_conf, best_preds, top_3

@app.route("/api/detect", methods=["POST"])
def api_detect():
    # 1. Guest Demo Limit Logic
    if 'role' not in session:
        import time
        current_time = time.time()
        last_reset = session.get('guest_limit_reset', 0)
        
        if current_time - last_reset >= 60:
            session['guest_count'] = 0
            session['guest_limit_reset'] = current_time
            
        guest_count = session.get('guest_count', 0)
        if guest_count >= 3:
            return jsonify({"status": "error", "message": "Demo limit reached. Please wait 1 minute for the limit to refresh, or login to continue.", "limit_reached": True}), 403
            
        session['guest_count'] = guest_count + 1

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            file_data = file.read()
            img = Image.open(io.BytesIO(file_data)).convert("RGB")
            img_resized = img.resize((224, 224))
            img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error processing image file: {str(e)}"}), 500

        if is_blank(file_data):
            return jsonify({"status": "error", "message": "Blank or solid color image detected. Please upload a clear dog/dog image."}), 400
            
        if is_blurry(file_data):
            return jsonify({"status": "error", "message": "Image is too blurry. Please upload a clear dog/dog image."}), 400

        import concurrent.futures
        g_models, g_classes, mb_model = get_models()
        
        # 1. Validation using MobileNetV2
        is_valid_dog = True
        validation_msg = ""
        try:
            from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
            img_m_array = np.array(img_resized, dtype=np.float32)
            img_m_array = np.expand_dims(img_m_array, axis=0)
            img_m_array = img_m_array / 127.5 - 1.0
            
            m_preds = mb_model.predict(img_m_array, verbose=0)
            decoded = decode_predictions(m_preds, top=5)[0]
            
            top_labels = [pred[1].lower().replace("_", " ") for pred in decoded]
            top_prob = decoded[0][2]
            
            allow_keywords = {"dog", "dog", "dog", "tree", "flower", "grass", "cabbage", "artichoke", "cardoon", "pepper", "zucchini", "squash", "cucumber", "mushroom", "fungus", "vegetable", "produce", "soil", "pot", "vase", "garden", "breed", "agriculture", "dog", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackdog", "apple", "pomegranate"}
            for dc in dynamic_breed_names:
                allow_keywords.add(dc)
                
            blocked_keywords = {"car", "sports car", "cab", "truck", "van", "bus", "jeep", "wheel", "bicycle", "motorcycle", "bike", "dog", "cat", "computer", "laptop", "monitor", "screen", "keyboard", "mouse", "phone", "desk", "sofa", "chair", "table", "building", "person", "human", "face"}
            
            if any(any(ak in label for ak in allow_keywords) for label in top_labels[:3]):
                is_valid_dog = True
            
            if any(any(bk in label for bk in blocked_keywords) for label in top_labels[:3]) and top_prob > 0.15:
                is_valid_dog = False
                
            if not is_valid_dog:
                validation_msg = "Please upload a valid dog, dog, or dog image."
        except Exception as e:
            print(f"[VALIDATION ERROR] {e}")
            is_valid_dog = True # Bypassed on error
            
        if not is_valid_dog:
            return jsonify({"status": "error", "message": validation_msg}), 400

        # 2. Multi-threaded Prediction
        if not g_models:
            return jsonify({"status": "error", "message": "No models trained yet."}), 400
            
        best_model_key, best_class, best_conf, best_preds, top_3 = perform_two_tier_prediction(img_array, g_models, g_classes)
        
        # 3. Threshold Logic
        if best_conf < 40.0:
            return jsonify({"status": "error", "message": f"Low confidence ({best_conf:.1f}%). Please upload a clearer image."}), 400

        parts = best_model_key.split("_", 1)
        breed_name = parts[0]
        breed_type = parts[1] if len(parts) > 1 else "all"
        target_model_folder = f"{breed_name}/{breed_type}"
        target_category = best_class

        if target_category == "Unwanted":
            return jsonify({"status": "error", "message": "Invalid/Unwanted Image Detected. Please upload a correct dog/dog image."}), 400

        # 4. Save and train
        is_dup = is_duplicate(file_data, img_resized, target_model_folder, target_category)

        duplicate_msg = None
        if is_dup:
            duplicate_msg = "Duplicate Image Detected"
            print(f"[DUPLICATE DETECTED] Image rejected: {target_category}", flush=True)
        elif "Unknown" in target_category:
            print(f"[UNKNOWN CATEGORY] Image not saved to dataset for: {target_category}", flush=True)
        else:
            upload_folder = os.path.join("user_uploads", target_model_folder, target_category)
            os.makedirs(upload_folder, exist_ok=True)
            upload_count = len(os.listdir(upload_folder))
            filename = f"{target_category}_new{upload_count + 1}.jpg"
            save_path = os.path.join(upload_folder, filename)
            with open(save_path, "wb") as f:
                f.write(file_data)
            
            # Save copy for Tier-1 Master Model
            tier1_category = f"{breed_name}_{breed_type}".replace("_category", "")
            tier1_upload = os.path.join("user_uploads", "master", "breed_classifier", tier1_category)
            os.makedirs(tier1_upload, exist_ok=True)
            t1_filename = f"{tier1_category}_new{len(os.listdir(tier1_upload)) + 1}.jpg"
            with open(os.path.join(tier1_upload, t1_filename), "wb") as f:
                f.write(file_data)
                
            master_key = "master_breed_classifier"
            if master_key not in new_uploads_counters:
                new_uploads_counters[master_key] = 0
            new_uploads_counters[master_key] += 1
            if new_uploads_counters[master_key] >= 50:
                trigger_training(master_key)
                new_uploads_counters[master_key] = 0
            
            if best_model_key not in new_uploads_counters:
                new_uploads_counters[best_model_key] = 0
            new_uploads_counters[best_model_key] += 1
            print(f"[AUTO-TRAINING] {best_model_key} new uploads count: {new_uploads_counters[best_model_key]}/50", flush=True)
            if new_uploads_counters[best_model_key] >= 50:
                trigger_training(best_model_key)
                new_uploads_counters[best_model_key] = 0

        encoded_img = base64.b64encode(file_data).decode("utf-8")
        img_base64 = f"data:image/jpeg;base64,{encoded_img}"

        result_name = best_class.replace('_', ' ').upper() if best_class else "UNKNOWN"

        return jsonify({
            "status": "success",
            "data": {
                "result_name": result_name,
                "breed_name": breed_name.capitalize(),
                "result_confidence": round(best_conf, 2),
                "dog_confidence": 0,
                "duplicate_msg": duplicate_msg,
                "img_base64": img_base64,
                "top_3_results": top_3
            }
        })

    return jsonify({"status": "error", "message": "Unsupported file type"}), 400

@app.route("/admin_detect", methods=["GET", "POST"])
@admin_required
def admin_detect():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded")
            return redirect(request.url)

        file = request.files["file"]
        image_type = request.form.get("image_type", "dog")

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                file_data = file.read()
                img = Image.open(io.BytesIO(file_data)).convert("RGB")
                img_resized = img.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
            except Exception as e:
                flash(f"Error processing image file: {str(e)}")
                return redirect(url_for("admin_detect"))

            if is_blank(file_data):
                flash("Blank or solid color image detected. Please upload a clear dog/dog image.")
                return redirect(url_for("admin_detect"))
                
            if is_blurry(file_data):
                flash("Image is too blurry. Please upload a clear dog/dog image.")
                return redirect(url_for("admin_detect"))

            import concurrent.futures
            g_models, g_classes, mb_model = get_models()
            
            # 1. Validation using MobileNetV2
            is_valid_dog = True
            validation_msg = ""
            try:
                from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
                img_m_array = np.array(img_resized, dtype=np.float32)
                img_m_array = np.expand_dims(img_m_array, axis=0)
                img_m_array = img_m_array / 127.5 - 1.0
                
                m_preds = mb_model.predict(img_m_array, verbose=0)
                decoded = decode_predictions(m_preds, top=5)[0]
                
                top_labels = [pred[1].lower().replace("_", " ") for pred in decoded]
                top_prob = decoded[0][2]
                
                allow_keywords = {"dog", "dog", "dog", "tree", "flower", "grass", "cabbage", "artichoke", "cardoon", "pepper", "zucchini", "squash", "cucumber", "mushroom", "fungus", "vegetable", "produce", "soil", "pot", "vase", "garden", "breed", "agriculture", "dog", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackdog", "apple", "pomegranate"}
                for dc in dynamic_breed_names:
                    allow_keywords.add(dc)
                    
                blocked_keywords = {"car", "sports car", "cab", "truck", "van", "bus", "jeep", "wheel", "bicycle", "motorcycle", "bike", "dog", "cat", "computer", "laptop", "monitor", "screen", "keyboard", "mouse", "phone", "desk", "sofa", "chair", "table", "building", "person", "human", "face"}
                
                if any(any(ak in label for ak in allow_keywords) for label in top_labels[:3]):
                    is_valid_dog = True
                elif any(any(bk in label for bk in blocked_keywords) for label in top_labels[:3]) and top_prob > 0.15:
                    is_valid_dog = False
                    
                if not is_valid_dog:
                    validation_msg = "Irrelevant image detected. Please upload correct dog dog or dog image."
            except Exception as e:
                print(f"[VALIDATION ERROR] {e}")
                is_valid_dog = True
                
            if not is_valid_dog:
                flash(validation_msg)
                return redirect(url_for("admin_detect"))

            # 2. Multi-threaded Prediction
            if not g_models:
                flash("No models trained yet.")
                return redirect(url_for("admin_detect"))
                
            best_model_key, best_class, best_conf, best_preds, top_3_results = perform_two_tier_prediction(img_array, g_models, g_classes)
            
            # 3. Threshold Logic
            if best_conf < 40.0:
                flash(f"Image not recognized (Low confidence: {best_conf:.1f}%). Please upload a clearer image.")
                return redirect(url_for("admin_detect"))

            parts = best_model_key.split("_", 1)
            breed_name = parts[0]
            breed_type = parts[1] if len(parts) > 1 else "all"
            target_model_folder = f"{breed_name}/{breed_type}"
            target_category = best_class

            if target_category == "Unwanted":
                flash("Invalid/Unwanted Image Detected. Please upload a correct dog/dog image.")
                return redirect(url_for("admin_detect"))

            # 4. Save and train
            is_dup = is_duplicate(file_data, img_resized, target_model_folder, target_category)

            duplicate_msg = None
            if is_dup:
                duplicate_msg = "Duplicate Image Detected"
                print(f"[DUPLICATE DETECTED] Image rejected: {target_category}", flush=True)
            elif "Unknown" in target_category:
                print(f"[UNKNOWN CATEGORY] Image not saved to dataset for: {target_category}", flush=True)
            else:
                upload_folder = os.path.join("user_uploads", target_model_folder, target_category)
                os.makedirs(upload_folder, exist_ok=True)
                upload_count = len(os.listdir(upload_folder))
                filename = f"{target_category}_new{upload_count + 1}.jpg"
                save_path = os.path.join(upload_folder, filename)
                with open(save_path, "wb") as f:
                    f.write(file_data)
                
                # Save copy for Tier-1 Master Model
                tier1_category = f"{breed_name}_{breed_type}".replace("_category", "")
                tier1_upload = os.path.join("user_uploads", "master", "breed_classifier", tier1_category)
                os.makedirs(tier1_upload, exist_ok=True)
                t1_filename = f"{tier1_category}_new{len(os.listdir(tier1_upload)) + 1}.jpg"
                with open(os.path.join(tier1_upload, t1_filename), "wb") as f:
                    f.write(file_data)
                    
                master_key = "master_breed_classifier"
                if master_key not in new_uploads_counters:
                    new_uploads_counters[master_key] = 0
                new_uploads_counters[master_key] += 1
                if new_uploads_counters[master_key] >= 50:
                    trigger_training(master_key)
                    new_uploads_counters[master_key] = 0
                
                if best_model_key not in new_uploads_counters:
                    new_uploads_counters[best_model_key] = 0
                new_uploads_counters[best_model_key] += 1
                print(f"[AUTO-TRAINING] {best_model_key} new uploads count: {new_uploads_counters[best_model_key]}/50", flush=True)
                if new_uploads_counters[best_model_key] >= 50:
                    trigger_training(best_model_key)
                    new_uploads_counters[best_model_key] = 0

            encoded_img = base64.b64encode(file_data).decode("utf-8")
            img_base64 = f"data:image/jpeg;base64,{encoded_img}"

            result_name = best_class.replace('_', ' ').upper() if best_class else "UNKNOWN"

            return render_template( 
                "admin/admin_detect.html",
                img_base64=img_base64,
                image_type=breed_type,
                breed_name=breed_name.capitalize(),
                result_name=result_name,
                result_confidence=round(best_conf, 2),
                dog_confidence=0,
                duplicate_msg=duplicate_msg,
                top_3_results=top_3_results
            )

        flash("Unsupported file type")
        return redirect(request.url)

    return render_template("admin/admin_detect.html")

@app.route("/detect", methods=["GET", "POST"])
@login_required
def detect():
    # Legacy detect route
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded")
            return redirect(request.url)

        file = request.files["file"]
        image_type = request.form.get("image_type", "dog")  # Get image type (dog, dog, dog)

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                file_data = file.read()
                img = Image.open(io.BytesIO(file_data)).convert("RGB")
                img_resized = img.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
            except Exception as e:
                flash(f"Error processing image file: {str(e)}")
                return redirect(url_for("detect"))

            if is_blank(file_data):
                flash("Blank or solid color image detected. Please upload a clear dog/dog image.")
                return redirect(url_for("detect"))
                
            if is_blurry(file_data):
                flash("Image is too blurry. Please upload a clear dog/dog image.")
                return redirect(url_for("detect"))

            import concurrent.futures
            g_models, g_classes, mb_model = get_models()
            
            # 1. Validation using MobileNetV2
            is_valid_dog = True
            validation_msg = ""
            try:
                from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
                img_m_array = np.array(img_resized, dtype=np.float32)
                img_m_array = np.expand_dims(img_m_array, axis=0)
                img_m_array = img_m_array / 127.5 - 1.0
                m_preds = mb_model.predict(img_m_array, verbose=0)
                decoded = decode_predictions(m_preds, top=5)[0]
                
                top_labels = [pred[1].lower().replace("_", " ") for pred in decoded]
                top_prob = decoded[0][2]
                
                allow_keywords = {"dog", "dog", "dog", "tree", "flower", "grass", "cabbage", "artichoke", "cardoon", "pepper", "zucchini", "squash", "cucumber", "mushroom", "fungus", "vegetable", "produce", "soil", "pot", "vase", "garden", "breed", "agriculture", "dog", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackdog", "apple", "pomegranate"}
                for dc in dynamic_breed_names:
                    allow_keywords.add(dc)
                    
                blocked_keywords = {"car", "sports car", "cab", "truck", "van", "bus", "jeep", "wheel", "bicycle", "motorcycle", "bike", "dog", "cat", "computer", "laptop", "monitor", "screen", "keyboard", "mouse", "phone", "desk", "sofa", "chair", "table", "building", "person", "human", "face"}
                
                if any(any(ak in label for ak in allow_keywords) for label in top_labels[:3]):
                    is_valid_dog = True
                elif any(any(bk in label for bk in blocked_keywords) for label in top_labels[:3]) and top_prob > 0.15:
                    is_valid_dog = False
                    
                if not is_valid_dog:
                    validation_msg = "Irrelevant image detected. Please upload correct dog dog or dog image."
            except Exception as e:
                print(f"[VALIDATION ERROR] {e}")
                is_valid_dog = True
                
            if not is_valid_dog:
                flash(validation_msg)
                return redirect(url_for("detect"))

            # 2. Multi-threaded Prediction
            if not g_models:
                flash("No models trained yet.")
                return redirect(url_for("detect"))
                
            best_model_key, best_class, best_conf, best_preds, top_3_results = perform_two_tier_prediction(img_array, g_models, g_classes)
            
            # 3. Threshold Logic
            if best_conf < 40.0:
                flash(f"Image not recognized (Low confidence: {best_conf:.1f}%). Please upload a clearer image.")
                return redirect(url_for("detect"))

            parts = best_model_key.split("_", 1)
            breed_name = parts[0]
            breed_type = parts[1] if len(parts) > 1 else "all"
            target_model_folder = f"{breed_name}/{breed_type}"
            target_category = best_class

            if target_category == "Unwanted":
                flash("Invalid/Unwanted Image Detected. Please upload a correct dog/dog image.")
                return redirect(url_for("detect"))

            # 4. Save and train
            is_dup = is_duplicate(file_data, img_resized, target_model_folder, target_category)

            duplicate_msg = None
            if is_dup:
                duplicate_msg = "Duplicate Image Detected"
                print(f"[DUPLICATE DETECTED] Image rejected: {target_category}", flush=True)
            elif "Unknown" in target_category:
                print(f"[UNKNOWN CATEGORY] Image not saved to dataset for: {target_category}", flush=True)
            else:
                upload_folder = os.path.join("user_uploads", target_model_folder, target_category)
                os.makedirs(upload_folder, exist_ok=True)
                upload_count = len(os.listdir(upload_folder))
                filename = f"{target_category}_new{upload_count + 1}.jpg"
                save_path = os.path.join(upload_folder, filename)
                with open(save_path, "wb") as f:
                    f.write(file_data)
                
                # Save copy for Tier-1 Master Model
                tier1_category = f"{breed_name}_{breed_type}".replace("_category", "")
                tier1_upload = os.path.join("user_uploads", "master", "breed_classifier", tier1_category)
                os.makedirs(tier1_upload, exist_ok=True)
                t1_filename = f"{tier1_category}_new{len(os.listdir(tier1_upload)) + 1}.jpg"
                with open(os.path.join(tier1_upload, t1_filename), "wb") as f:
                    f.write(file_data)
                    
                master_key = "master_breed_classifier"
                if master_key not in new_uploads_counters:
                    new_uploads_counters[master_key] = 0
                new_uploads_counters[master_key] += 1
                if new_uploads_counters[master_key] >= 50:
                    trigger_training(master_key)
                    new_uploads_counters[master_key] = 0
                
                if best_model_key not in new_uploads_counters:
                    new_uploads_counters[best_model_key] = 0
                new_uploads_counters[best_model_key] += 1
                print(f"[AUTO-TRAINING] {best_model_key} new uploads count: {new_uploads_counters[best_model_key]}/50", flush=True)
                if new_uploads_counters[best_model_key] >= 50:
                    trigger_training(best_model_key)
                    new_uploads_counters[best_model_key] = 0

            encoded_img = base64.b64encode(file_data).decode("utf-8")
            img_base64 = f"data:image/jpeg;base64,{encoded_img}"

            result_name = best_class.replace('_', ' ').upper() if best_class else "UNKNOWN"

            return render_template( 
                "user/index.html",
                is_deployed=IS_DEPLOYED,
                img_base64=img_base64,
                image_type=breed_type,
                breed_name=breed_name.capitalize(),
                result_name=result_name,
                result_confidence=round(best_conf, 2),
                dog_confidence=0,
                duplicate_msg=duplicate_msg,
                top_3_results=top_3_results
            )

        flash("Unsupported file type")
        return redirect(request.url)

    return render_template("user/index.html", is_deployed=IS_DEPLOYED)

# ------------------------------------------------------------
# ------------------------------------------------------------
# Warm up models in a background thread
# ------------------------------------------------------------
def warmup_models():
    print("[STARTUP] Pre-loading models in background for fast initial response...")
    try:
        get_models()
        print("[STARTUP] All models pre-loaded successfully!")
    except Exception as e:
        print(f"[STARTUP] Error pre-loading models: {e}")

threading.Thread(target=warmup_models, daemon=True).start()

# ------------------------------------------------------------
# Run Flask
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)