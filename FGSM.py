# 对抗噪声算法FGSM

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


model = ...  # 加载模型
model.eval()

# 定义数据集和数据加载器
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root='path_to_your_dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 定义FGSM攻击
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# 生成对抗样本
epsilon = 0.1  # 设定扰动强度
adversarial_examples = []

for data, target in dataloader:
    data.requires_grad = True
    output = model(data)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = fgsm_attack(data, epsilon, data_grad)
    adversarial_examples.append((perturbed_data, target))

# 将对抗样本保存到新的数据集中
adversarial_dataset = [(img, label) for img, label in adversarial_examples]
torch.save(adversarial_dataset, 'path_to_save_adversarial_dataset.pt')


# 加载对抗样本数据集
adversarial_dataset = torch.load('path_to_save_adversarial_dataset.pt')
adversarial_loader = DataLoader(adversarial_dataset, batch_size=1, shuffle=True)

correct = 0
total = 0

for data, target in adversarial_loader:
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    total += target.size(0)

print(f'Accuracy on adversarial dataset: {correct / total * 100:.2f}%')


# 加载原始测试集
original_test_dataset = datasets.ImageFolder(root='path_to_your_test_dataset', transform=transform)
original_test_loader = DataLoader(original_test_dataset, batch_size=1, shuffle=True)

# 合并原始测试集和对抗样本数据集
combined_dataset = original_test_dataset + adversarial_dataset
combined_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True)

# 使用combined_loader进行模型测试
correct = 0
total = 0

for data, target in combined_loader:
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    total += target.size(0)

print(f'Accuracy on combined dataset: {correct / total * 100:.2f}%')
