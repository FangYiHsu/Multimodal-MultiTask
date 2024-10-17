from HPRNN_NAM_LeakyReLU import HPRNN
from MultiNet import *
from torchsummary import summary

obs_model_path = "model50_loss0.36020187107124574.pkl"
model_HPRNN_1 = HPRNN(64)
model_HPRNN_1.load_state_dict(torch.load(obs_model_path))
obs_hprnn = model_HPRNN_1.cuda()

model = MultiTaskModel(obs_hprnn).cuda()

# 計算總參數量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# 計算可訓練的參數量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {trainable_params}")