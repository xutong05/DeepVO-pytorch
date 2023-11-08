# DeepVO-pytorch

#### Tools/Methods: PyTorch, FlowNetSimple, LSTM, MSE, Weights & Biases

#### (1) Utilized rotation matrix and geometric equation to acquire the absolute and relative poses of various images

#### (2) Reproduced DeepVO network architectures by integrating FlowNetSimple pretrained model with LSTM

#### (3) Compiled loss function by using Mean Square Error (MSE) of all positions $`p`$ and orientations $`\varphi`$

#### (4) Evaluated the influence of epochs and hyperparameter $`k`$ in the loss function on the DeepVO network using KITTI and nuScenes datasets according to translation and rotation RMSE values

#### (5) Plotted trajectories on the predicted results from test sequences and compared them with original routes
