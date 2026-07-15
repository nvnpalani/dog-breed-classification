import os
import json
import re
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
import shutil
from datetime import datetime
import hashlib

# =========================
# UTILS
# =========================
def log_history(model_name, trigger_category, images_added, images_removed, old_acc, new_acc, status, dataset_count=0, dataset_hash=""):
    history_file = "app_data/training_history.json"
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except: pass

    last_version_num = 0
    for entry in history:
        if entry.get("model_name") == model_name and "version" in entry:
            v_str = entry["version"].replace("v", "")
            if v_str.isdigit():
                last_version_num = max(last_version_num, int(v_str))
    
    if status == "Success":
        new_version = f"v{last_version_num + 1}"
    else:
        new_version = f"v{last_version_num}" if last_version_num > 0 else "v0"
        
    history.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "category_trigger": trigger_category,
        "images_added": images_added,
        "images_removed": images_removed,
        "old_accuracy": round(old_acc, 2),
        "new_accuracy": round(new_acc, 2),
        "status": status,
        "dataset_count": dataset_count,
        "dataset_hash": dataset_hash,
        "version": new_version
    })
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)
        
def get_last_training_info(model_name):
    history_file = "app_data/training_history.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
                for entry in reversed(history):
                    if entry.get("model_name") == model_name and entry.get("status") == "Success":
                        return entry.get("dataset_count", 0), entry.get("dataset_hash", ""), entry.get("new_accuracy", 0.0), entry.get("version", "v0")
        except: pass
    return 0, "", 0.0, "v0"

def get_dataset_stats(dataset_path, upload_path=""):
    count = 0
    hasher = hashlib.md5()
    supported_formats = (".jpg", ".jpeg", ".png", ".webp")
    
    all_files = []
    for path in [dataset_path, upload_path]:
        if path and os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(supported_formats):
                        filepath = os.path.join(root, f)
                        try:
                            stat = os.stat(filepath)
                            all_files.append(f"{f}_{stat.st_size}_{stat.st_mtime}")
                            count += 1
                        except: pass
                        
    all_files.sort()
    for item in all_files:
        hasher.update(item.encode('utf-8'))
        
    return count, hasher.hexdigest()

def check_dataset_balance(dataset_path, upload_path):
    class_counts = {}
    
    if os.path.exists(dataset_path):
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
                class_counts[item] = len(files)
                
    if os.path.exists(upload_path):
        for item in os.listdir(upload_path):
            item_path = os.path.join(upload_path, item)
            if os.path.isdir(item_path):
                files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
                class_counts[item] = class_counts.get(item, 0) + len(files)
            
    if not class_counts: return True, "", class_counts
    
    print("\n--- Current Class Image Counts ---")
    for cat, count in sorted(class_counts.items()):
        print(f"  - {cat}: {count} images")
    print("----------------------------------")
    
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    
    if min_count == 0: 
        return False, f"Class with 0 images found. Minimum required is at least 1, and ideally matching the max count ({max_count}).", class_counts
    
    ratio = max_count / min_count
    if ratio > 1.2:
        required_min = int(max_count / 1.2)
        return False, f"Max class has {max_count} images, Min class has {min_count}. Ratio is {ratio:.2f} (Limit is 1.2). Min class needs at least {required_min} images to train!", class_counts
        
    return True, "", class_counts

# =========================
# CLEAN DATASET
# =========================
def clean_dataset(dataset_path):
    if not os.path.exists(dataset_path): return 0
    print(f"\nChecking images in {dataset_path}")
    supported_formats = (".jpg", ".jpeg", ".png", ".webp")
    removed_count = 0

    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.lower().endswith(supported_formats): continue

            try:
                cv_img = cv2.imread(file_path)
                if cv_img is None:
                    raise Exception("Could not read image with cv2")
                
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if variance < 30.0:
                    print(f"Removing blurry image (variance {variance:.2f}):", file_path)
                    os.remove(file_path)
                    removed_count += 1
                    continue

                img = Image.open(file_path)
                if img.mode != "RGB" or img.size != (224, 224):
                    img = img.convert("RGB")
                    img = img.resize((224, 224))
                    img.save(file_path)
            except Exception as e:
                print("Removing corrupted image:", file_path)
                try:
                    os.remove(file_path)
                    removed_count += 1
                except: pass
                
        if root != dataset_path and not os.listdir(root):
            try:
                os.rmdir(root)
                print(f"Removed empty folder: {root}")
            except: pass

    print(f"Dataset cleaned successfully! Removed {removed_count} images.")
    return removed_count

# =========================
# TRAIN MODEL
# =========================

def smart_rename(folder_path):
    if not os.path.exists(folder_path):
        return
        
    supported_formats = (".jpg", ".jpeg", ".png", ".webp")
    renamed_total = 0
    
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        prefix = class_name.replace(" ", "_")
        pattern = re.compile(f"^{re.escape(prefix)}_(\\d+)\\.([a-zA-Z0-9]+)$", re.IGNORECASE)
        
        existing_indices = set()
        incorrect_files = []
        
        # 1. Identify files
        for f in os.listdir(class_path):
            f_path = os.path.join(class_path, f)
            if not os.path.isfile(f_path):
                continue
                
            if not f.lower().endswith(supported_formats):
                continue
                
            match = pattern.match(f)
            if match:
                idx = int(match.group(1))
                existing_indices.add(idx)
            else:
                incorrect_files.append(f)
                
        if not incorrect_files:
            continue
            
        # 2. Find next available index
        next_idx = 1
        
        # 3. Rename incorrect files
        for bad_f in incorrect_files:
            while next_idx in existing_indices:
                next_idx += 1
                
            ext = bad_f.split('.')[-1].lower()
            new_name = f"{prefix}_{next_idx}.{ext}"
            
            old_path = os.path.join(class_path, bad_f)
            new_path = os.path.join(class_path, new_name)
            
            try:
                os.rename(old_path, new_path)
                existing_indices.add(next_idx)
                renamed_total += 1
            except Exception as e:
                print(f"Error renaming {bad_f}: {e}")
                
    if renamed_total > 0:
        print(f"[SMART RENAME] Renamed {renamed_total} images in {folder_path} to match sequential order.")


def train_model(dataset_path, upload_path, model_file, class_file, model_name):
    # 0. Smart Rename
    print(f"\n--- Smart Renaming Dataset for {model_name} ---")
    smart_rename(dataset_path)
    smart_rename(upload_path)

    last_count, last_hash, old_acc, last_version = get_last_training_info(model_name)
    current_count, current_hash = get_dataset_stats(dataset_path, upload_path)
    
    print(f"\n--- Checking Retraining Condition for {model_name} ---")
    print(f"Last Trained Count: {last_count} | Hash: {last_hash}")
    print(f"Current Dataset Count: {current_count} | Hash: {current_hash}")
    
    if last_count > 0 and last_count == current_count and last_hash == current_hash:
        print("\nDo NOT Retrain. Dataset Unchanged.")
        return
    else:
        print("\nDataset Modified. Proceeding with Quality Check, Balance Check, and Retraining.")

    # 1. Clean datasets
    removed1 = clean_dataset(dataset_path)
    removed2 = clean_dataset(upload_path)
    total_removed = removed1 + removed2
    
    # 2. Check balance and get counts
    is_balanced, reason, class_counts = check_dataset_balance(dataset_path, upload_path)
    if not is_balanced:
        print(f"\n[WARNING] Dataset Imbalanced for {model_name}")
        print(f"Reason: {reason}")
        print("Proceeding anyway to ensure continuous learning with AUTO CLASS WEIGHTS...\n")
        
    # 3. Move images temporarily
    moved_files = []
    if os.path.exists(upload_path):
        for root, dirs, files in os.walk(upload_path):
            for f in files:
                src_file = os.path.join(root, f)
                rel_path = os.path.relpath(root, upload_path)
                dst_dir = os.path.join(dataset_path, rel_path)
                os.makedirs(dst_dir, exist_ok=True)
                dst_file = os.path.join(dst_dir, f)
                shutil.move(src_file, dst_file)
                moved_files.append((src_file, dst_file))

    print(f"\nTraining Dataset: {dataset_path}")

    try:
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(224, 224),
            batch_size=8
        )

        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(224, 224),
            batch_size=8
        )

        class_names = train_dataset.class_names
        print("\nClasses:", class_names)
        
        with open(class_file, "w") as f:
            json.dump(class_names, f)

        AUTOTUNE = tf.data.AUTOTUNE
        
        # === Dynamic Scenario Augmentation ===
        shadow_missing = False
        rotation_needed = False
        zoom_needed = False
        
        try:
            if os.path.exists("app_data/scenario_cache.json"):
                with open("app_data/scenario_cache.json", "r") as f:
                    cache = json.load(f)
                    model_key = "dog"
                    if "dog" in model_name.lower(): model_key = "dog"
                    elif "dog" in model_name.lower(): model_key = "dog"
                    
                    if model_key in cache:
                        cats = cache[model_key].get("categories", [])
                        for c in cats:
                            name = c.get("name", "")
                            pct = c.get("percentage", 100)
                            if "Shadow" in name and pct < 50: shadow_missing = True
                            if ("Tilted" in name or "Side" in name) and pct < 50: rotation_needed = True
                            if ("Far" in name or "Close" in name) and pct < 50: zoom_needed = True
        except Exception as e:
            pass

        print("\n--- Dynamic Augmentation Strategy ---")
        aug_layers = [layers.RandomFlip("horizontal_and_vertical")]
        if rotation_needed: 
            aug_layers.append(layers.RandomRotation(0.3))
            print("[+] Added Aggressive Rotation (Missing Views)")
        else:
            aug_layers.append(layers.RandomRotation(0.1))
            
        if zoom_needed:
            aug_layers.append(layers.RandomZoom(0.2))
            print("[+] Added Random Zoom (Missing Distances)")
            
        if shadow_missing:
            aug_layers.append(layers.RandomBrightness(factor=(-0.4, 0.1)))
            print("[+] Added Random Darkening (Missing Shadows)")
            
        data_augmentation = tf.keras.Sequential(aug_layers)
        
        train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        # =====================================
        
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        class EpochSaverCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                with open("last_epoch.txt", "w") as f:
                    f.write(str(epoch + 1))
        
        epochs_to_run = 100
        checkpoint_file = model_file.replace('.keras', '_checkpoint.keras')

        initial_epoch_num = 0
        if os.path.exists(checkpoint_file):
            if os.path.exists("last_epoch.txt"):
                try:
                    with open("last_epoch.txt", "r") as f:
                        initial_epoch_num = int(f.read().strip())
                except:
                    pass
            print(f"\n[INFO] Found CHECKPOINT '{checkpoint_file}'. Resuming training from Epoch {initial_epoch_num + 1}...")
            model = tf.keras.models.load_model(checkpoint_file, custom_objects={"preprocess_input": preprocess_input})
        elif os.path.exists(model_file):
            print(f"\n[INFO] Found existing model '{model_file}'. Loading for FINE-TUNING...")
            model = tf.keras.models.load_model(model_file, custom_objects={"preprocess_input": preprocess_input})
            tf.keras.backend.set_value(model.optimizer.learning_rate, 1e-5)
        else:
            print(f"\n[INFO] No existing model found. Training from SCRATCH...")
            base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            for layer in base_model.layers[-20:]:
                layer.trainable = True

            model = models.Sequential([
                layers.Lambda(preprocess_input),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.2),
                layers.RandomContrast(0.2),
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(len(class_names), activation="softmax")
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

        print(f"\n[INFO] Auto-Training initiated: Max Epochs = {epochs_to_run}")
        
        early_stop = EarlyStopping(
            monitor='val_accuracy', 
            patience=3,
            min_delta=0.005,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint_cb = ModelCheckpoint(
            filepath=checkpoint_file,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        lr_reduce = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        # Auto-Calculate Class Weights
        class_weights_dict = {}
        total_images = sum(class_counts.values())
        num_classes = len(class_names)
        
        # Estimate image counts based on the 80-20 split
        train_img_count = int(total_images * 0.8)
        val_img_count = total_images - train_img_count

        print(f"\n==================================================")
        print(f" MODEL TRAINING SUMMARY ")
        print(f"==================================================")
        print(f" Breed Name       : {model_name.replace(' Model', '')}")
        print(f" Total Images    : {total_images}")
        print(f" Train Images    : ~{train_img_count}")
        print(f" Valid Images    : ~{val_img_count}")
        print(f" Total Classes   : {num_classes}")
        print(f" Target Epochs   : {epochs_to_run}")
        print(f"==================================================\n")
        
        print("\n--- Applying Auto Class Weights ---")
        for idx, class_name in enumerate(class_names):
            count = class_counts.get(class_name, 1) # Fallback to 1 to prevent division by zero
            weight = total_images / (num_classes * count)
            class_weights_dict[idx] = weight
            print(f"  - {class_name}: weight={weight:.2f}")

        history = model.fit(
            train_dataset, 
            validation_data=validation_dataset, 
            epochs=epochs_to_run,
            initial_epoch=initial_epoch_num,
            callbacks=[early_stop, lr_reduce, checkpoint_cb, EpochSaverCallback()],
            class_weight=class_weights_dict,
            verbose=1
        )

        # 4. Accuracy Protection
        val_acc = history.history['val_accuracy'][-1] * 100
        # old_acc already populated by get_last_training_info
        images_added = len(moved_files)
        
        dataset_name = os.path.basename(dataset_path)
        print(f"\n{dataset_name} total accuracy {val_acc:.0f}/100")

        print("\n--- Sub-folder (Class-wise) Accuracy ---")
        class_correct = {c: 0 for c in class_names}
        class_total = {c: 0 for c in class_names}

        for images, labels in validation_dataset:
            preds = model.predict(images, verbose=0)
            pred_labels = tf.argmax(preds, axis=1)
            for true_label, pred_label in zip(labels.numpy(), pred_labels.numpy()):
                class_name = class_names[true_label]
                class_total[class_name] += 1
                if true_label == pred_label: class_correct[class_name] += 1

        for class_name in class_names:
            if class_total[class_name] > 0:
                acc = (class_correct[class_name] / class_total[class_name]) * 100
                print(f"  - {class_name}: {acc:.0f}/100")
            else:
                print(f"  - {class_name}: N/A (No validation images)")
        print("----------------------------------------\n")

        # Check if accuracy improved or if this is the first training run
        final_count, final_hash = get_dataset_stats(dataset_path, "")

        if old_acc == 0.0 or val_acc >= (old_acc - 5.0):
            status_msg = "Improved" if val_acc >= old_acc else "Acceptable Drop"
            print(f"[SUCCESS] Model {status_msg} ({old_acc:.2f}% -> {val_acc:.2f}%). Merging Dataset.")
            model.save(model_file)
            log_history(model_name, "Auto", images_added, total_removed, old_acc, val_acc, "Success", dataset_count=final_count, dataset_hash=final_hash)
            
            # Clear upload path
            if os.path.exists(upload_path):
                shutil.rmtree(upload_path)
                os.makedirs(upload_path, exist_ok=True)
                
            print(f"\nModel Saved: {model_file}")
            print(f"Classes Saved: {class_file}")
        else:
            print(f"[REJECTED] Model Degraded ({old_acc:.2f}% -> {val_acc:.2f}%). Reverting Dataset.")
            for src, dst in moved_files:
                os.makedirs(os.path.dirname(src), exist_ok=True)
                shutil.move(dst, src)
            log_history(model_name, "Auto", 0, total_removed, old_acc, val_acc, "Rejected", dataset_count=last_count, dataset_hash=last_hash)

        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
            except: pass
        if os.path.exists("last_epoch.txt"):
            try:
                os.remove("last_epoch.txt")
            except: pass

    except Exception as e:
        print(f"Training crashed: {e}")
        # Revert on crash
        for src, dst in moved_files:
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.move(dst, src)


if __name__ == "__main__":
    import sys
    import os
    
    base_breed_dir = "dog_datasets"
    base_upload_dir = "user_uploads"

    if os.path.exists(base_breed_dir):
        dataset_path = base_breed_dir
        upload_path = base_upload_dir
        
        # Ensure model directory exists
        model_dir = os.path.join("models", "dog")
        os.makedirs(model_dir, exist_ok=True)
        
        # Standardized model and class file names
        model_file = os.path.join(model_dir, "breeds_model.keras")
        class_file = os.path.join(model_dir, "breeds_classes.json")
        model_name_log = "Dog Breeds Model"
            
        print(f"\n--- STARTING TRAINING FOR ALL DOG BREEDS ---")
        train_model(
            dataset_path=dataset_path,
            upload_path=upload_path,
            model_file=model_file,
            class_file=class_file,
            model_name=model_name_log
        )