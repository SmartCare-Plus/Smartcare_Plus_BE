import os
import pyperclip

# --- CONFIGURATION ---
# Directories to skip entirely (Removes Flutter boilerplate and ML temp folders)
EXCLUDED_DIRS = {
    'node_modules', '.git', '.vscode', '.idea', '__pycache__', 
    'venv', '.venv', 'env', 'build', 'ios', 'android', 'linux', 'macos', 'windows',
    '.dart_tool', '.pytest_cache', 'dist', 'out', 'logs', 'runs',
    'debug', 'release', 'migrations', '.github'
}

# Extensions to skip (Audio, Binaries, Data, Documents)
EXCLUDED_EXTS = {
    # Media & Audio
    '.mp3', '.wav', '.flac', '.ogg', '.m4a',
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.mp4', '.mov', '.pdf',
    # ML Models, Data & Databases
    '.h5', '.pkl', '.onnx', '.pt', '.pth', '.tflite', '.pb', '.csv', '.xlsx', 
    '.db', '.sqlite', '.sqlite3', '.cache',
    # Documents & IDEs
    '.docx', '.doc', '.pptx', '.iml', '.xml',
    # Compressed/Binaries
    '.zip', '.tar', '.gz', '.7z', '.exe', '.dll', '.so', '.dylib', '.pyc',
    # Others
    '.txt', '.log', '.lock', '.yaml.lock', '.pubxml', '.gitattributes'
}

# Files we definitely WANT to keep
WHITELIST_FILES = {'pubspec.yaml', 'package.json', 'Dockerfile', 'requirements.txt', '.gitignore'}

def should_skip(path, filename):
    # 1. Skip if file is in an excluded directory
    parts = path.split(os.sep)
    if any(d in EXCLUDED_DIRS for d in parts):
        return True
            
    # 2. Keep if specifically whitelisted
    if filename in WHITELIST_FILES:
        return False
        
    # 3. Skip by extension
    ext = os.path.splitext(filename)[1].lower()
    if ext in EXCLUDED_EXTS:
        return True
        
    return False

def collect_code(root_dir):
    all_content = []
    
    for root, dirs, files in os.walk(root_dir):
        # Modify dirs in place to stop os.walk from even entering these folders
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        for file in files:
            if not should_skip(root, file):
                # Extra check: Skip the script itself
                if file == os.path.basename(__file__):
                    continue
                    
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_dir)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                        if content: # Only add if file isn't empty
                            header = f"\n{'='*60}\nFILE: {relative_path}\n{'='*60}\n"
                            all_content.append(header + content)
                            print(f"✅ Added: {relative_path}")
                except Exception as e:
                    print(f"❌ Failed to read {relative_path}: {e}")

    return "\n".join(all_content)

if __name__ == "__main__":
    project_path = "." 
    print("🚀 Gathering cleaned code files (excluding audio, binaries, and boilerplate)...")
    
    final_string = collect_code(project_path)
    
    if final_string:
        pyperclip.copy(final_string)
        print("\n" + "—" * 40)
        print(f"✨ SUCCESS! Cleaned code copied to clipboard.")
        print(f"Total length: {len(final_string)} characters.")
        print("—" * 40)
    else:
        print("⚠️ No valid code files found.")