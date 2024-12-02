import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
    
from models.load import load_model
from models.HDGCN import Model
from training_functions import load_optimizer, train, evaluate

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parsed_args = parser.parse_args()
    selected_model = parsed_args.model

    # Handle the selected model
    if selected_model.lower() == "hdgcn":
        print("Model found")
        # Configuration parameters
        config = {
            'model_args': {
                'num_class': 5,
                'num_point': 25,
                'num_person': 1,
                'graph': 'Graph',
                'graph_args': {
                    'labeling_mode': 'spatial',
                    'CoM': 21,
                },
            },
            'weights': None,
            'ignore_weights': [],
            'base_lr': 0.1,
            'step': [20, 40, 60],
            'device_ids': [0],
            'optimizer_type': 'SGD',
            'nesterov': True,
            'batch_size': 64,
            'test_batch_size': 64,
            'start_epoch': 0,
            'num_epoch': 45,
            'weight_decay': 0.0004,
            'warm_up_epoch': 5,
            'loss_type': 'CE',
            'lr_ratio': 0.001,
            'lr_decay_rate': 0.1,
            'n_frames': 64,
            'n_joints': 25,
        }

        # Load data
        # ENSURE THAT THIS FILES EXISTS
        x_data = np.load("JointExtractor/output_dataset/final_joints.npy")
        y_data = np.load("JointExtractor/output_dataset/final_labels.npy")
    else:
        print("Model not found")
        exit()

    # Preprocess data
    x_data = preprocess_data(x_data, config['n_frames'], config['n_joints'])

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.30, random_state=42
    )

    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        x_train, x_test, y_train, y_test, config['batch_size']
    )

    # Instantiate model
    model = load_model(Model, config['model_args'])

    # Prepare optimizer
    optimizer = load_optimizer(
        optimizer_type=config['optimizer_type'],
        model=model,
        base_lr=config['base_lr'],
        weight_decay=config['weight_decay'],
        nesterov=config['nesterov']
    )

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"You are using {device}")

    # Move model to device
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Train the model
    train_model(model, optimizer, train_loader, test_loader, config, loss_fn, device)

def preprocess_data(x_data, n_frames, n_joints):
    n_samples = x_data.shape[0]

    for idx in range(n_samples):
        for frame in range(n_frames):
            joint_21 = x_data[idx, frame, 20]
            d_norm = np.sum((joint_21 - x_data[idx, frame, 0]) ** 2)
            d_norm = d_norm if d_norm != 0 else 1

            # Centering the joints
            x_data[idx, frame] -= joint_21
            x_data[idx, frame] /= d_norm

            if np.all(x_data[idx, frame] == 0):
                for prev_frame in range(frame - 1, -1, -1):
                    if not np.all(x_data[idx, prev_frame] == 0):
                        x_data[idx, frame] = x_data[idx, prev_frame]
                        break
    return x_data

def create_dataloaders(x_train, x_test, y_train, y_test, batch_size):
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # (N, C, T, V, M)
    x_train = x_train.permute(0, 3, 1, 2).unsqueeze(-1)
    x_test = x_test.permute(0, 3, 1, 2).unsqueeze(-1)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(model, optimizer, train_loader, test_loader, config, loss_fn, device):
    best_acc = 0
    num_epoch = config['num_epoch']
    base_lr = config['base_lr']
    warm_up_epoch = config['warm_up_epoch']
    lr_ratio = config['lr_ratio']

    for epoch in range(num_epoch):
        train(
            model, optimizer, train_loader, epoch, num_epoch,
            base_lr, warm_up_epoch, lr_ratio, loss_fn, device
        )
        acc = evaluate(model, test_loader, loss_fn, device)
        if acc > best_acc:
            best_acc = acc
            # Save the best model
            torch.save(
                model.state_dict(),
                f'output/hdgcn_model_epoch_{epoch+1}_beta.pt'
            )
        print(f'Best Accuracy: {best_acc * 100:.2f}%')

if __name__ == "__main__":
    main()
