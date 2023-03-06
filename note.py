import torch 

model = torch.jit.load('and_model.pt')

sample = [torch.randn((3,640,480))]
output = model(sample)
print(output)
print('exit success')