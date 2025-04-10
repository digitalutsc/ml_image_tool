# This script is used before CNN model training and QA steps.
# By selecting every nth image (or image pairs), it gives a 
# homogeneous sample of the original image set.
#
# Before running this script, make sure that if pairs exist, then a pair of
# images should have the same filename except that the right version of it has a '_r' suffix.

import os
import shutil

src_root = 'C:\\D\\DSU\\Kanagaratnam\\Train 10th - Copy\\900 CCW'
dst_folder = 'C:\\D\\DSU\\github\\test_sample'

lst = os.listdir(src_root)

def create_sample(src_root, dst_folder, every_nth, mode):
    counter = 0
    print(counter)

    # Create the destination folder if it doesn't exist
    os.makedirs(dst_folder, exist_ok=True)

    # Walk through all directories and files in the source root
    for root, _, files in os.walk(src_root):
        for file in files:
            nm, ext = os.path.splitext(file)

            # Check for pairs: file.ext with file_r.ext or vice versa
            pair_file = None
            if nm + '_r' + ext in lst:
                pair_file = nm + '_r' + ext
            elif '_r' in nm and nm.replace('_r', '') + ext in lst:
                pair_file = nm.replace('_r', '') + ext

            if pair_file:
                counter += 1

                # Construct full file paths
                src_file1 = os.path.join(root, file)
                dst_file1 = os.path.join(dst_folder, file)
                src_file2 = os.path.join(root, pair_file)
                dst_file2 = os.path.join(dst_folder, pair_file)

                # Process every nth pair
                if counter % every_nth == 0:
                    try:
                        if mode == 'copy':
                            shutil.copy2(src_file1, dst_file1)
                            shutil.copy2(src_file2, dst_file2)
                            print(f"Copied {src_file1} to {dst_file1}")
                            print(f"Copied {src_file2} to {dst_file2}")
                        elif mode == 'move':
                            shutil.move(src_file1, dst_file1)
                            shutil.move(src_file2, dst_file2)
                            print(f"Moved {src_file1} to {dst_file1}")
                            print(f"Moved {src_file2} to {dst_file2}")
                        else:
                            print(f"Invalid mode: {mode}")
                            return
                    except Exception as e:
                        print(f"Error processing files {file} and {pair_file}: {e}")
            else:
                counter += 1

                # Construct full file paths
                src_file1 = os.path.join(root, file)
                dst_file1 = os.path.join(dst_folder, file)

                # Process every nth pair
                if counter % every_nth == 0:
                    try:
                        if mode == 'copy':
                            shutil.copy2(src_file1, dst_file1)
                            print(f"Copied {src_file1} to {dst_file1}")
                        elif mode == 'move':
                            shutil.move(src_file1, dst_file1)
                            print(f"Moved {src_file1} to {dst_file1}")
                        else:
                            print(f"Invalid mode: {mode}")
                            return
                    except Exception as e:
                        print(f"Error processing files {file} and {pair_file}: {e}")


# Call the function
create_sample(src_root, dst_folder, 20, 'copy')