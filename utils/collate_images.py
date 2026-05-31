import os
import shutil
from tqdm import tqdm

def consolidate_images(source_root, destination_folder):
    """
    Moves all images from subfolders into one destination folder.
    Renames files if a collision occurs to prevent overwriting.
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")

    # Supported image extensions
    valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
    
    # Get list of all files to process
    files_to_move = []
    for root, _, files in os.walk(source_root):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_ext:
                files_to_move.append(os.path.join(root, file))

    # Move files with progress bar
    for file_path in tqdm(files_to_move, desc="Consolidating images"):
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(destination_folder, file_name)

        # Handle filename collisions
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(file_name)
            counter = 1
            new_name = f"{name}_{counter}{ext}"
            while os.path.exists(os.path.join(destination_folder, new_name)):
                counter += 1
                new_name = f"{name}_{counter}{ext}"
            dest_path = os.path.join(destination_folder, new_name)

        shutil.move(file_path, dest_path)

    print(f"\nSuccessfully moved files to: {destination_folder}")

if __name__ == "__main__":
    # Update these paths to match your directory structure
    SOURCE_DIRECTORY = r"C:\Users\wings\sevenwingsInc\01_clients\26-03_convFormer_ML+CNN\convFormer-model\steatosis"
    DESTINATION_DIRECTORY = r"C:\Users\wings\sevenwingsInc\01_clients\26-03_convFormer_ML+CNN\convFormer-model\steatosis_images"

    consolidate_images(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)

