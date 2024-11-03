import os
import cv2
import mediapipe as mp
import numpy as np
import argparse
from tqdm import tqdm
import re

parser = argparse.ArgumentParser()
parser.add_argument("class_path")
args = parser.parse_args()
data_path = args.class_path
pattern = r".*/(.+)$"

# Extract with Regex
clase = re.search(pattern,data_path).group(1)
categorias_elegidas = [clase]

def mediapipe_to_ntu(landmarks):
    ntu_joints = np.zeros((26,3))
    # Convert Mediapipe landmarks to numpy array
    landmarks = np.array(landmarks)
    # NTU Joint 1 - Base of the spine
    ntu_joints[1] = (landmarks[23] + landmarks[24]) / 2
    # NTU Joint 3 - Neck
    ntu_joints[3] = (landmarks[9] + landmarks[10]) / 2
    # NTU Joint 4 - Head
    ntu_joints[4] = (landmarks[5] + landmarks[2]) / 2
    # NTU Joint 5 - Left shoulder
    ntu_joints[5] = landmarks[11]
    # NTU Joint 6 - Left elbow
    ntu_joints[6] = landmarks[13]
    # NTU Joint 7 - Left wrist
    ntu_joints[7] = landmarks[15]
    # NTU Joint 8 - Tip of the right hand
    ntu_joints[8] = (landmarks[19] + landmarks[15] + landmarks[17])/3
    # NTU Joint 9 - Right shoulder
    ntu_joints[9] = landmarks[12]
    # NTU Joint 10 - Right elbow
    ntu_joints[10] = landmarks[14]
    # NTU Joint 11 - Right wrist
    ntu_joints[11] = landmarks[16]
    # NTU Joint 12 - Tip of the right hand
    ntu_joints[12] = (landmarks[20] + landmarks[18] + landmarks[16])/3
    # NTU Joint 13 - Left hip
    ntu_joints[13] = landmarks[23]
    # NTU Joint 14 - Left knee
    ntu_joints[14] = landmarks[25]
    # NTU Joint 15 - Left ankle
    ntu_joints[15] = landmarks[27]
    # NTU Joint 16 - Left foot
    ntu_joints[16] = landmarks[31]
    # NTU Joint 17 - Right hip
    ntu_joints[17] = landmarks[24]
    # NTU Joint 18 - Right knee
    ntu_joints[18] = landmarks[26]
    # NTU Joint 19 - Right ankle
    ntu_joints[19] = landmarks[28]
    # NTU Joint 20 - Right foot
    ntu_joints[20] = landmarks[32]
    # NTU Joint 21 - Spine
    ntu_joints[21] = (landmarks[12] + landmarks[11]) / 2
    # NTU Joint 2 - Middle of the spine
    ntu_joints[2] = (ntu_joints[21] + ntu_joints[1]) / 2
    # NTU Joint 22 - right index
    ntu_joints[22] = landmarks[19]
    # NTU Joint 23 - Left thumb
    ntu_joints[23] = landmarks[21]
    # NTU Joint 24 - 
    ntu_joints[24] = landmarks[20]
    # NTU Joint 25 - Right thumb
    ntu_joints[25]  = landmarks[22]
    return ntu_joints[1:]

def process_video(video_path, pose):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            joints = mediapipe_to_ntu(landmarks)
            frames.append(joints)
        else:
            frames.append(np.zeros((25, 3)))
    cap.release()
    # Ajusta los frames para tener exactamente 64
    num_frames = len(frames)
    if num_frames > 64:
        indices = np.linspace(0, num_frames - 1, 64).astype(int)
        frames = [frames[i] for i in indices]
    elif num_frames < 64:
        frames.extend([frames[-1]] * (64 - num_frames))
    return frames

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    # Inicializa listas para datos y etiquetas  
    data = []
    labels = []
    print("proccessing videos...")
    # Procesa cada clase con una barra de progreso
    for video_file in tqdm(os.listdir(data_path), desc=f"Procesando {clase}", unit="video"):
        video_path = os.path.join(data_path, video_file)
        frames = process_video(video_path, pose)
        data.append(frames)
        labels.append(clase)
    data = np.array(data)  
    labels = np.array(labels)
    # Save
    np.save(f'output_dataset/data_{clase}.npy', data)
    np.save(f'output_dataset/labels_{clase}.npy', labels)

if __name__ == "__main__":
    main()


