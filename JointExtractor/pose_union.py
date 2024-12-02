import os
import numpy as np

def consolidate_files(input_dir, output_data_file, output_labels_file):
    all_data = []
    all_labels = []
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if file_name.startswith("data_") and file_name.endswith(".npy"):
            print(f"Loading data from {file_name}")
            data = np.load(file_path)
            all_data.append(data)
        elif file_name.startswith("labels_") and file_name.endswith(".npy"):
            print(f"Loading labels from {file_name}")
            labels = np.load(file_path)
            all_labels.append(labels)
    
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Save the consolidated data and labels
    np.save(output_data_file, all_data)
    np.save(output_labels_file, all_labels)
    print(f"Union complete. Saved data to {output_data_file} and labels to {output_labels_file}")

input_directory = "output_dataset"
final_data_file = "output_dataset/final_joint.npy"
final_labels_file = "output_dataset/final_labels.npy"

consolidate_files(input_directory, final_data_file, final_labels_file)
