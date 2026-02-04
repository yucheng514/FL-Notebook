# 导入所需的第三方库
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import argparse
import copy
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 导入项目的模块
from federated.partition import cifar_iid, cifar_noniid

# step 0）定义一些超参数
def args_parser():
    # 1. 创建一个解析器对象 (造一个控制面板)
    parser = argparse.ArgumentParser(description='FedAvg 联邦学习实验设置')

    # 2. 添加参数 (给面板上加按钮)
    # 格式：parser.add_argument('参数名', 类型, 默认值, 说明)

    parser.add_argument('--epochs', type=int, default=10, help='的轮数')
    parser.add_argument('--num_users', type=int, default=100, help='总共有多少个客户端')
    parser.add_argument('--frac', type=float, default=0.1, help='每轮选中客户端的比例 (0.1 表示 10%)')
    parser.add_argument('--lr', type=float, default=0.01, help='客户端本地训练的学习率')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='分类任务的类别数')
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet分布参数，越小越不均匀')
    # 【新增】FedProx 的核心参数 mu
    # 当 mu = 0 时，就是标准的 FedAvg
    # 当 mu > 0 时 (比如 0.01, 0.1, 1), 就是 FedProx
    parser.add_argument('--mu', type=float, default=0.0, help='FedProx parameter mu (0 for FedAvg)')
    # 3. 开始解析 (读取你在终端敲的命令)
    args = parser.parse_args()
    return args
args = args_parser()

# step 1）检测当前运行环境的操作系统和硬件信息；
def get_device():
    # 1. 优先检查是否有 NVIDIA 显卡 (云端环境)
    if torch.cuda.is_available():
        print(f"检测到 NVIDIA 显卡: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    # 2. 检查是否是 Mac (M1/M2/M3/M4) 的 MPS 加速
    elif torch.backends.mps.is_available():
        print("检测到 Apple Silicon (M系列芯片)，启用 MPS 加速")
        return torch.device("mps")

    # 3. 都没有就用 CPU (保底)
    else:
        print("未检测到 GPU，使用 CPU")
        return torch.device("cpu")


args.device = get_device()
# 例子：
# model = MyModel().to(device)
# data, target = data.to(device), target.to(device)

# step 2） 转Tensor，归一化，下载 CIFAR10 数据集（训练集和测试集）
# 1. 定义数据预处理 (转为 Tensor, 归一化)
# 训练集：必须加“随机裁剪”和“水平翻转”，这是 ResNet 的标配
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),       # <--- 新增：随机裁剪
    transforms.RandomHorizontalFlip(),          # <--- 新增：随机左右翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 测试集：保持原样，只做标准化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 2. 下载并加载训练集
# root='./dataset' 表示把数据下载到当前目录下的 dataset 文件夹里
trainset = torchvision.datasets.CIFAR10(
    root='./dataset',
    train=True,
    download=True,  # <--- 关键参数：如果没有数据，它会自动下载；如果有，它就跳过
    transform=train_transform
)

# 3. 下载并加载测试集，并iid划分用户数据
testset = torchvision.datasets.CIFAR10(
    root='./dataset',
    train=False,
    download=True,
    transform=test_transform
)
print("CIFAR10下载/加载完成！")

if args.iid:
    dict_users = cifar_iid(trainset, args.num_users)
else:
    dict_users = cifar_noniid(trainset, args.num_users, alpha=args.alpha)
img_size = trainset[0][0].shape


# step 3）定义一个 CNN 模型用于 CIFAR10 分类
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# net_glob = CNNCifar(args=args).to(args.device)
# print('CNNCifar模型样子：\n {}'.format(net_glob))
def build_resnet18(args):
    # 使用 torchvision 的 ResNet-18
    net = torchvision.models.resnet18(weights=None)  # 不加载预训练权重（本地先跑通/预训练可控）

    # CIFAR-10: 32x32，调整 stem
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()

    # 调整分类头
    net.fc = nn.Linear(net.fc.in_features, int(args.num_classes))
    return net

# 替换原来的 CNNCifar 实例化
net_glob = build_resnet18(args).to(args.device)
# print("ResNet-18 模型样子：\n {}".format(net_glob))


net_glob.train()

# 返回模型的所有可学习参数的字典。
w_glob = net_glob.state_dict()
# print(w_glob)



# training
loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []
loss_test_history = []  # <--- 新增：记录测试 Loss
acc_test_history = []   # <--- 新增：记录测试准确率

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
class LocalUpdate():
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # 【FedProx 新增步骤 1】
        # 在训练开始前，深拷贝一份“全局模型”作为参照物 (Anchor)
        # 因为 net 刚传进来时就是全局模型，所以直接考一份锁住即可
        if self.args.mu > 0:
            global_net = copy.deepcopy(net)
            global_net.eval()
            for param in global_net.parameters():
                param.requires_grad = False  # 锁死参数，不参与梯度计算


        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                # 【FedProx 新增步骤 2】
                # 如果 mu > 0，计算 Proximal Term 并加到 Loss 上
                if self.args.mu > 0:
                    proximal_term = 0.0
                    # 遍历当前模型 (net) 和 锁死的全局模型 (global_net) 的参数
                    for w, w_t in zip(net.parameters(), global_net.parameters()):
                        # 计算 L2 距离的平方: ||w - w_t||^2
                        proximal_term += (w - w_t).norm(2) ** 2

                    # 修改总 Loss：原始Loss + (mu / 2) * 距离
                    loss += (self.args.mu / 2) * proximal_term

                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def test_model(net_g, dataset, args):
    """
    专门用来测试全局模型的函数
    """
    net_g.eval()  # <--- 关键！开启“考试模式”，关掉 Dropout，固定 BN 层

    # 建立测试数据加载器
    test_loader = DataLoader(dataset, batch_size=args.bs, shuffle=False)

    test_loss = 0
    correct = 0

    # 不计算梯度，节省显存，加速计算
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)

            output = net_g(data)

            # 计算总 Loss (Sum up batch loss)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            # 计算预测正确的个数
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataset)
    accuracy = 100.00 * correct / len(dataset)

    return accuracy, test_loss
for iter in range(args.epochs):
    net_glob.train()  # <--- 新增：每轮开始前，确保切回“训练模式”
    loss_locals = []
    w_locals = []
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False) # 每轮只训练一部分的客户端

    for idx in idxs_users:
        local = LocalUpdate(args=args, dataset=trainset, idxs=dict_users[idx])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        if args.all_clients:
            w_locals[idx] = copy.deepcopy(w)
        else:
            w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
    # update global weights


    w_glob = FedAvg(w_locals)
    net_glob.load_state_dict(w_glob)
    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    loss_train.append(loss_avg)

    acc_test, loss_test = test_model(net_glob, testset, args)
    loss_test_history.append(loss_test)
    acc_test_history.append(acc_test)

    # Average loss：当前这一轮（Current Round）”所有参与训练的客户端的“平均训练Loss”。
    # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    print('Round {:3d}, Train Loss {:.3f} | Test Loss {:.3f} | Test Acc: {:.2f}%'.format(
        iter, loss_avg, loss_test, acc_test))

# plot loss curve
plt.figure()
plt.plot(range(len(loss_train)), loss_train)
plt.ylabel('train_loss')
save_dir = './save'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
timestamp = datetime.now().astimezone().strftime('%Y%m%d_%H:%M')
plt.savefig(os.path.join(save_dir,'fed_{}_{}_{}_C{}_iid{}_{}.png'.format('cifar', 'cnn', args.epochs, args.frac, args.iid, timestamp)))

# testing
def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            # data, target = data.cuda(), target.cuda()
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss
net_glob.eval()
acc_train, loss_train = test_img(net_glob, trainset, args)
acc_test, loss_test = test_img(net_glob, testset, args)
print("Training accuracy: {:.2f}".format(acc_train))
print("Testing accuracy: {:.2f}".format(acc_test))

