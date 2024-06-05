import torch 
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.loader import DataLoader

def prep_data(data_pt_list,f_split = 0.8,target = 'homo', isvalid=True, f_split_valid=0.2,random_state_valid=42):
    train_dataset_list = []
    test_dataset_list = []
    for data_pt in data_pt_list:
        dataset1 = torch.load(data_pt)
        for data in dataset1:
            data.y = data[target]
        train_dataset1 = Subset(dataset1,range(0,int(len(dataset1)*f_split)))
        test_dataset1 = Subset(dataset1,range(int(len(dataset1)*f_split),len(dataset1)))
        train_dataset_list.append(train_dataset1)
        test_dataset_list.append(test_dataset1)
    
    print(f'training set sizes: {[len(train_dataset1) for train_dataset1 in train_dataset_list]}',flush=True)
    train_dataset = ConcatDataset(train_dataset_list)
    test_dataset = ConcatDataset(test_dataset_list)
    if isvalid:
        train_dataset, valid_dataset = train_test_split(train_dataset,test_size=f_split_valid,shuffle=True,random_state=random_state_valid)
        return train_dataset, valid_dataset, test_dataset
    else:
        return train_dataset, test_dataset

def prep_data_random(data_pt_list,f_split = 0.8, isvalid=True, f_split_valid=0.2,random_state_valid=42):
    train_dataset_list = []
    test_dataset_list = []
    for data_pt in data_pt_list:
        dataset1 = torch.load(data_pt)
        target = 'homo'
        for data in dataset1:
            data.y = data[target]
        train_dataset1, test_dataset1 = train_test_split(dataset1, test_size=1-f_split, shuffle=True)
        train_dataset_list.append(train_dataset1)
        test_dataset_list.append(test_dataset1)
    
    print(f'training set sizes: {[len(train_dataset1) for train_dataset1 in train_dataset_list]}',flush=True)
    train_dataset = ConcatDataset(train_dataset_list)
    test_dataset = ConcatDataset(test_dataset_list)
    if isvalid:
        train_dataset, valid_dataset = train_test_split(train_dataset,test_size=f_split_valid,shuffle=True,random_state=random_state_valid)
        return train_dataset, valid_dataset, test_dataset
    else:
        return train_dataset, test_dataset

# def output_last_layer(model,test_dataset,batch_size,device):
#     model = model.to(device)


def eval_test(checkpoint_file,model,test_dataset,batch_size,device):

    checkpoint = torch.load(checkpoint_file)  # replace with your checkpoint's path
    model.load_state_dict(checkpoint['model_state_dict'])

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss_func = torch.nn.L1Loss()

    total_loss = 0
    all_preds = []
    all_targets = []

    for data in loader_test:
        data = data.to(device)
        with torch.no_grad():
            output = model(data).squeeze()
        loss = loss_func(output, data.y.squeeze())
        total_loss += loss.item() * len(data.y)
        all_preds.append(output.cpu().numpy())
        all_targets.append(data.y.cpu().numpy())

    average_loss = total_loss / len(test_dataset)
    print(f"Average L1 Loss on Test Dataset: {average_loss:.4f}")

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    # Compute R^2 value
    from sklearn.metrics import r2_score
    r2 = r2_score(all_targets, all_preds)
    print(f"R^2 Score on Test Dataset: {r2:.4f}")

    import matplotlib.pyplot as plt
    np.savetxt('test_predict.txt',np.column_stack([np.array(all_targets).squeeze(),np.array(all_preds)]),header='Target Prediction')
    plt.figure(figsize=(4, 3),dpi=300)
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([np.min(all_targets), np.max(all_targets)], [np.min(all_targets), np.max(all_targets)], color='red', linestyle='--')  # a reference line for perfect prediction
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Prediction vs Actual - R^2: {r2:.4f}")
    plt.tight_layout()
    plt.savefig('./performance.png')
 
    return average_loss, r2
