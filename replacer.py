import os
import re

def replace_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return

    new_content = content
    
    # Specific replacements
    replacements = {
        "dog_datasets": "dog_datasets",
        "Dog Breed Identification": "Dog Breed Identification",
        "Dog Breed Identifier": "Dog Breed Identifier",
        "Dog Breed": "Dog Breed",
        "dog_image": "dog_image",
        "dog": "dog",
        "Dog": "Dog",
        "dynamic_breed_names": "dynamic_breed_names",
        "breed_name": "breed_name",
        "Breed Name": "Breed Name",
        "add_breed.py": "add_breed.py",
        "breed_type": "breed_type",
        "Breed Type": "Breed Type",
        "dog_category": "dog_category",
        "dog_category": "dog_category",
        "dog_types": "dog_types",
        "dog": "dog",
        "Dog": "Dog",
        "dog": "dog",
        "Dog": "Dog",
        "dog": "dog",
        "Dog": "Dog",
        "Total Breeds": "Total Breeds",
        "Breed Models": "Breed Models",
        "breed": "breed",
        "Breed": "Breed",
        "category": "category",
        "Category": "Category"
    }

    # Iterate over replacements. We must be careful about order.
    # Replace longer phrases first.
    sorted_replacements = sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True)
    
    for old, new in sorted_replacements:
        # Simple text replace.
        new_content = new_content.replace(old, new)
        
    if content != new_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

def main():
    base_dir = r"d:\dog breed identification"
    
    # Files and extensions to process
    exts = ('.py', '.html', '.js', '.css', '.txt')
    
    for root, dirs, files in os.walk(base_dir):
        if 'node_modules' in root or '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith(exts):
                filepath = os.path.join(root, file)
                replace_in_file(filepath)
                
    # Rename add_breed.py to add_breed.py
    old_add_breed = os.path.join(base_dir, "add_breed.py")
    new_add_breed = os.path.join(base_dir, "add_breed.py")
    if os.path.exists(old_add_breed):
        os.rename(old_add_breed, new_add_breed)
        print(f"Renamed {old_add_breed} to {new_add_breed}")

    # Rename dog_datasets to dog_datasets if exists
    old_ds = os.path.join(base_dir, "dog_datasets")
    new_ds = os.path.join(base_dir, "dog_datasets")
    if os.path.exists(old_ds) and not os.path.exists(new_ds):
        os.rename(old_ds, new_ds)
        print(f"Renamed {old_ds} to {new_ds}")

if __name__ == "__main__":
    main()
