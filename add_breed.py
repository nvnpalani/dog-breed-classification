import os
import sys

def create_breed_structure(breed_name):
    breed_name = breed_name.lower().strip()
    
    # Base directories
    base_dirs = ["dog_datasets", "user_uploads"]
    
    # Subdirectories to create
    sub_dirs = [
        "dog_category",
        "dog_category",
        f"{breed_name}_types"
    ]
    
    print(f"Initializing folder structure for breed: '{breed_name.capitalize()}'...\n")
    
    created_count = 0
    for base in base_dirs:
        for sub in sub_dirs:
            target_path = os.path.join(base, breed_name, sub)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
                print(f"[+] Created: {target_path}")
                created_count += 1
            else:
                print(f"[-] Already exists: {target_path}")
                
    if created_count > 0:
        print(f"\n[SUCCESS] Successfully initialized {created_count} new folders for '{breed_name.capitalize()}'!")
    else:
        print(f"\n[INFO] Folders for '{breed_name.capitalize()}' already exist.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_breed.py <breed_name>")
        print("Example: python add_breed.py tomato")
        sys.exit(1)
        
    breed_input = sys.argv[1]
    create_breed_structure(breed_input)
