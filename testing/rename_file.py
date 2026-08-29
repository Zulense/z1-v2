import os
import glob

def clean_physical_files():
    # Use the absolute paths to be safe, regardless of where you run the script
    folders = [
        "Data/video_data",
        # "/nlsasfs/home/hfgenai/mansav/Data/video_text_latent" # Added this so your latents match!
    ]

    print("Renaming physical files on the hard drive...")
    
    for folder in folders:
        # Find ALL files in the directory
        all_files = glob.glob(f"{folder}/*")
        
        for old_path in all_files:
            dirname = os.path.dirname(old_path)
            old_filename = os.path.basename(old_path)
            
            # Apply all requested replacements to the filename
            new_filename = old_filename.replace("\uff5c", "_")
            new_filename = new_filename.replace("--", "_")
            new_filename = new_filename.replace(" ", "_")
            
            # Clean up any messy consecutive underscores
            while "__" in new_filename:
                new_filename = new_filename.replace("__", "_")
                
            # Rename only if the filename actually changed
            if old_filename != new_filename:
                new_path = os.path.join(dirname, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {new_filename}")
                except Exception as e:
                    print(f"Error renaming {old_filename}: {e}")

    print("\nFinished renaming all files!")

if __name__ == "__main__":
    clean_physical_files()