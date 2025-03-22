import subprocess
import os
import time

def run_script(script_name, description):
    print(f"\n{'='*50}")
    print(f"Running {description}...")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(['python', script_name], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully completed {script_name}")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"✗ Error running {script_name}")
            if result.stderr:
                print("Error message:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Failed to run {script_name}: {str(e)}")
        return False
        
    return True

def main():
    start_time = time.time()
    
    print("\nSquat Analysis Pipeline")
    print("----------------------")
    
    # Check if required files exist
    required_files = {
        'process_squat.py': 'Script to process model video',
        'process_user_input.py': 'Script to process user video',
        'compare.py': 'Script to compare videos'
    }
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("\n✗ Error: Missing required files:")
        for file in missing_files:
            print(f"  - {file}: {required_files[file]}")
        return
    
    # Step 1: Generate model video
    if not run_script('process_squat.py', 'Model Video Generation'):
        print("\n✗ Pipeline stopped due to error in model video generation")
        return
        
    # Step 2: Generate user video
    if not run_script('process_user_input.py', 'User Video Generation'):
        print("\n✗ Pipeline stopped due to error in user video generation")
        return
        
    # Step 3: Run comparison
    if not run_script('compare.py', 'Video Comparison'):
        print("\n✗ Pipeline stopped due to error in video comparison")
        return
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*50)
    print("Pipeline Complete!")
    print(f"Total time: {total_time:.1f} seconds")
    print("="*50)
    
    # Check output files
    output_files = ['SquatLabeledSide.mp4', 'UserInputLabeled.mp4']
    print("\nChecking output files:")
    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
            print(f"✓ {file} ({size:.1f} MB)")
        else:
            print(f"✗ {file} not found")

if __name__ == "__main__":
    main() 