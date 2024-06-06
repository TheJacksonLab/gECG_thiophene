import torch 
from sklearn.model_selection import train_test_split
from dig.threedgraph.evaluation import ThreeDEvaluator
import os
from src.model_gECG import gECG, run_modified
from src.tools import prep_data,eval_test

train_dataset, test_dataset = prep_data(['../data/processed/dftdata_CG3R_L6.pt',
                                         ],isvalid=False,f_split=0.9) # f_split gives the 

loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()
run3d = run_modified()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_dataset, valid_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
model = gECG(cutoff=5.0, num_layers=3, hidden_channels=256, 
            middle_channels=128, out_channels=1, num_radial=6, 
            num_spherical=3, num_output_layers=4)
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
        epochs=10, batch_size=32, vt_batch_size=16, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15,save_dir='./save_train/',log_dir='./save_train/',is_test=False)

MAE, r2 = eval_test(model,test_dataset,batch_size=256,device=device,checkpoint_file="./save_train/valid_checkpoint.pt")
os.system('rm -r ./save_train')  # Clean up the directory after each run

