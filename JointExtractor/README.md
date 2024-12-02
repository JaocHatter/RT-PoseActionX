# How to Use the Joint Extractor

The **Joint Extractor** processes videos located in the `input_dataset` directory to extract joint information. It follows these steps:

1. **Pose Estimation:** Uses MediaPipe to obtain pose data.
2. **Topology Conversion:** Converts the MediaPipe topology (33 points) to the NTU topology (25 points). This conversion is essential for compatibility with models like **HD-GCN**, which require the NTU format.

## Processing Modes: Parallel or Sequential

You can choose to process the data using either **parallel** or **sequential** methods. 

- **Parallel Processing:** Utilizes multiple CPU cores (requires Rust).
- **Sequential Processing:** Processes data one at a time (does not require Rust).

### Parallel Processing
To use parallel processing, follow these steps:

```bash
# Build and execute the Rust program
cargo run 

# Condense all output files into a single dataset
python3 pose_union.py
```

### Sequential Processing
For sequential processing, use the following commands:

```bash
# Extract poses sequentially
python3 pose_extractor_seq.py

# Condense all output files into a single dataset
python3 pose_union.py
```

## Output
After processing, you will obtain an `output_dataset` directory. Ensure that your model script correctly links to the generated files to avoid errors during training.

### Verifying the Output in `main.py`
Check that the file paths in your `main.py` script point to the generated dataset:

```python
# Line 56 of main.py
x_data = np.load("JointExtractor/output_dataset/final_joints.npy")
y_data = np.load("JointExtractor/output_dataset/final_labels.npy")
```
