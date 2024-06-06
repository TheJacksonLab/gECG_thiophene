
import torch 
import numpy as np
from sklearn.model_selection import train_test_split
from dig.threedgraph.evaluation import ThreeDEvaluator
from src.model_gECG import gECG, run_modified
from src.tools import eval_test, predict

def load_model(device):
    model = gECG(cutoff=5.0, num_layers=3, hidden_channels=256, 
                middle_channels=128, out_channels=1, num_radial=6, 
                num_spherical=3, num_output_layers=4)
    checkpoint = torch.load('../models/model_DFT_CG3R_checkpoint.pt',map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

test_dataset = torch.load('../data/processed/dftdata_CG3R_L6.pt') 

loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()
run3d = run_modified()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = load_model(device)

output = predict(model,test_dataset,32,device)
np.savetxt('output.txt',output)

MAE, r2 = eval_test(model,test_dataset,batch_size=32,device=device)
