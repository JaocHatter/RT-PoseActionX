import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from models.HDGCN import Model
from models.load import load_model

# Verifiquemos el uso de la GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

# Inicializar Mediapipe Pose
mp_pose = mp.solutions.pose

model_args = {
    'num_class': 5,
    'num_point': 25,
    'num_person': 1,
    'graph': 'Graph',
    'graph_args': {
        'labeling_mode': 'spatial',
        'CoM': 21,
    },
}

# Cargar el modelo pre-entrenado
hd_gcn_model = load_model(Model, model_args, "output/hdgcn_model_epoch_12_beta.pt", ignore_weights=[], device=device)
hd_gcn_model.to(device)
hd_gcn_model.eval()

# Lista de clases
clases = ["Punch", "JumpRope", "JumpingJack", "PullUps", "PushUps"]

# Función para mapear los puntos de Mediapipe a las articulaciones de NTU
def mediapipe_to_ntu(lms, image_shape):
    ntu_joints = np.zeros((25, 3))
    # Convertir los landmarks de Mediapipe a un array de numpy
    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in lms])
    # NTU Joint 1 - Base of the spine
    ntu_joints[0] = (landmarks[23] + landmarks[24]) / 2 
    # NTU Joint 3 - Neck
    ntu_joints[2] = (landmarks[9] + landmarks[10]) / 2
    # NTU Joint 4 - Head
    ntu_joints[3] = (landmarks[5] + landmarks[2]) / 2
    # NTU Joint 5 - Left shoulder
    ntu_joints[4] = landmarks[11]
    # NTU Joint 6 - Left elbow
    ntu_joints[5] = landmarks[13]
    # NTU Joint 7 - Left wrist
    ntu_joints[6] = landmarks[15]
    # NTU Joint 8 - Tip of the right hand
    ntu_joints[7] = (landmarks[19] + landmarks[15] + landmarks[17])/3
    # NTU Joint 9 - Right shoulder
    ntu_joints[8] = landmarks[12]
    # NTU Joint 10 - Right elbow
    ntu_joints[9] = landmarks[14]
    # NTU Joint 11 - Right wrist
    ntu_joints[10] = landmarks[16]
    # NTU Joint 12 - Tip of the right hand
    ntu_joints[11] = (landmarks[20] + landmarks[18] + landmarks[16])/3
    # NTU Joint 13 - Left hip
    ntu_joints[12] = landmarks[23]
    # NTU Joint 14 - Left knee
    ntu_joints[13] = landmarks[25]
    # NTU Joint 15 - Left ankle
    ntu_joints[14] = landmarks[27]
    # NTU Joint 16 - Left foot
    ntu_joints[15] = landmarks[31]
    # NTU Joint 17 - Right hip
    ntu_joints[16] = landmarks[24]
    # NTU Joint 18 - Right knee
    ntu_joints[17] = landmarks[26]
    # NTU Joint 19 - Right ankle
    ntu_joints[18] = landmarks[28]
    # NTU Joint 20 - Right foot
    ntu_joints[19] = landmarks[32]
    # NTU Joint 21 - Spine
    ntu_joints[20] = (landmarks[12] + landmarks[11]) / 2
    # NTU Joint 2 - Middle of the spine
    ntu_joints[1] = (ntu_joints[20] + ntu_joints[0]) / 2
    # NTU Joint 22 - right index
    ntu_joints[21] = landmarks[19]
    # NTU Joint 23 - Left thumb
    ntu_joints[22] = landmarks[21]
    # NTU Joint 24 - 
    ntu_joints[23] = landmarks[20]
    # NTU Joint 25 - Right thumb
    ntu_joints[24]  = landmarks[22]
    # Normalización centrada en la articulación 2 (spine)
    d_norm = ((ntu_joints[20]-ntu_joints[0])**2).sum()
    hdgcn_frame = (ntu_joints - ntu_joints[20])/d_norm #ntu_joints[1]
    return ntu_joints * np.array((image_shape[1],image_shape[0],1)), hdgcn_frame

# Función para dibujar el esqueleto NTU en la imagen
def draw_ntu_skeleton(image, joints):
    # Lista de huesos que conectan las articulaciones
    bones = (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
        (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
        (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
    )

    # Dibujar huesos
    for bone in bones:
        joint1 = joints[bone[0]-1]
        joint2 = joints[bone[1]-1]
        if joint1 is not None and joint2 is not None:
            x1, y1 = int(joint1[0]), int(joint1[1])
            x2, y2 = int(joint2[0]), int(joint2[1])
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(image, (x1, y1), 4, (0, 0, 255), -1)
            cv2.circle(image, (x2, y2), 4, (0, 0, 255), -1)
    return image

# Iniciar captura de video
cap = cv2.VideoCapture("videos/v_BoxingPunchingBag_g02_c01.avi")

if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_filename = 'demo/user_demo_1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Cola para almacenar los frames
frames_data = deque(maxlen=64)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("program isnt receiving frames, leaving for precaution...")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            ntu_j, hdgcn_frame = mediapipe_to_ntu(landmarks, frame.shape)
            frame = draw_ntu_skeleton(frame, ntu_j)
            frames_data.append(hdgcn_frame)

            if len(frames_data) == 64:
                frames_array = np.array(frames_data)
                frames_array = np.transpose(frames_array, (2, 0, 1))
                data = np.zeros((1, 3, 64, 25, 1))
                data[0, :, :, :, 0] = frames_array
                data_tensor = torch.from_numpy(data).float().to(device)

                with torch.no_grad():
                    outputs = torch.nn.functional.softmax(hd_gcn_model(data_tensor))
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    if outputs[0][predicted_class] > 0.7:
                        class_label = clases[predicted_class]
                    else: 
                        frames_data.clear()
                        class_label = '-'

                frames_data = deque(list(frames_data)[32:], maxlen=64)
                cv2.putText(frame, f'Action: {class_label}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f'Acción: {class_label}')
        else:
            frames_data.clear()

        # Mostrar el frame
        cv2.imshow('Real time human action recognition', frame)

        # Escribir el frame en el archivo de video
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
cap.release()
out.release()  # Cerrar el archivo de video
cv2.destroyAllWindows()
