from setuptools import setup, find_packages

# Install torch+cuda manually to avoid errors
setup(
    name="poseactionx",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "einops",
        "tqdm==4.66.5",
        "numpy==1.26.3",
        "scikit-learn==1.5.2",
        "opencv-python==4.10.0.84",
        "mediapipe==0.10.15"
    ],
    entry_points={
        "console_scripts": [
            "poseactionx=main:main",  
        ],
    },
)
