from efficientnet_pytorch_3d import EfficientNet3D
import torch
from torchsummary import summary

device = 'cpu'

model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=1)

summary(model, input_size=(1, 224, 224, 224))

model = model.to(device)
inputs = torch.randn((1, 1, 224, 224, 224)).to(device)
labels = torch.tensor([0]).to(device)
# test forward
num_classes = 2

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
for epoch in range(2):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    print('[%d] loss: %.3f' % (epoch + 1, loss.item()))

print('Finished Training')
