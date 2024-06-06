import torch 
import numpy as np
from sklearn.model_selection import train_test_split
from dig.threedgraph.evaluation import ThreeDEvaluator
from src.model_gECG import gECG, run_modified
from src.tools import prep_data, eval_test
import os 

def load_model(device):
    model = gECG(cutoff=5.0, num_layers=3, hidden_channels=256, 
                middle_channels=128, out_channels=1, num_radial=6, 
                num_spherical=3, num_output_layers=4)
    checkpoint = torch.load('../models/model_CG3R_checkpoint.pt',map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    #### freeze till the interaction layers
    count = 0
    for layer in model.children():
        count += 1
        if count <= 4:  
            for param in layer.parameters():
                param.requires_grad = False
            # print(layer)
        else:
            break
    return model

loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()
run3d = run_modified()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

useable_dataset_homo1, test_dataset_homo1 = prep_data(['../data/processed/dftdata_CG3R_L6.pt',
                                         '../data/processed/dftdata_CG3R_L10.pt',
                                         '../data/processed/dftdata_CG3R_L14.pt'],target='homo',isvalid=False)
num_data = len(useable_dataset_homo1)

# initialize the model 
model = load_model(device)

os.system('rm ./save_train_homo1/valid_checkpoint.pt')
train_dataset_homo1, valid_dataset_homo1 = train_test_split(useable_dataset_homo1,test_size=0.2,random_state=np.random.randint(1e6))
print(f'training dataset size = {len(train_dataset_homo1)}')

run3d.run(device, train_dataset_homo1, valid_dataset_homo1, test_dataset_homo1, model, loss_func, evaluation,
        epochs=50, batch_size=128, vt_batch_size=64, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15,save_dir='./save_train/',log_dir='./save_train/',is_test=False)

for layer in model.children():
    for param in layer.parameters():
        param.requires_grad = True

checkpoint = torch.load('./save_train/valid_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
run3d.run(device, train_dataset_homo1, valid_dataset_homo1, test_dataset_homo1, model, loss_func, evaluation,
        epochs=30, batch_size=128, vt_batch_size=64, lr=0.0001, lr_decay_factor=0.5, lr_decay_step_size=15,save_dir='./save_train/',log_dir='./save_train/',is_test=False)
MAE, r2 = eval_test(model,test_dataset_homo1,batch_size=64,device=device,checkpoint_file="./save_train/valid_checkpoint.pt")