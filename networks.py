# 1.前向传播
# 2.求损失
# 3.反向传播,目的----求的参数的梯度
# 4.参数优化,更新参数

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
# 优化器
import torch.optim as optim
# 什么是激活函数？
#     --激活函数就是将本来线性的替代加入非线性的转换,能够是拟合更加确切

# 有标签的数据叫做有监督模型，否则叫做无监督模型
# 这里的损失函数是怎么求的？参照的那个Y是什么？
# 因为只在训练阶段起作用,然而又是用的共同的环境,所以输入就是,状态空间+所有无人机可能的动作
class CriticNetwork(nn.Module):
    # 配置属性
    # beta: 学习
    # input_dims: 状态向量维度
    # fc1_dims, fc2_dims: 两层全连接隐藏层的维度
    # n_agents: 智能体数量
    # n_actions: 每个智能体的动作维度
    # name, chkpt_dir: 用于保存模型名称与目录
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        # 定义一个优化器,其实就是求梯度,就比如随机梯度下降,这里的Adam是一个优化器类型,相较于普通的SGD更快更稳定(就是wt=wt+....的功能，只不过实现的方法不一样)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # 更新学习率,学习率是不断更新的,不是一成不变的,这段代码就表示每5000步将学习率改变为原来的1/3
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.33)
        # 自动检测当前电脑有没有GPU（CUDA），然后设置训练设备为：
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # 把当前模型（self）中的所有参数和缓存数据都移动到你上面设置的设备上（GPU或CPU）。
        # 如果你不写这一句，模型默认在CPU上，哪怕你有GPU也不会自动使用。
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    # 加载或保存模型的参数
    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        # self.state_dict() 是一个包含模型参数的字典--是继承自父类的方法--nn.Module--
        # state_dict()默认只保存“模型中注册的所有参数”，不管它们是否需要优化，也就是说： 只要参数是模型的一部分（比如nn.Linear的权重和偏置），无论requires_grad = True
        # 还是False，它们都会被state_dict()保存下来。
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):#这不就是pai的函数形式吗，所以输出的就是某个状态下，每一个动作的概率
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        # 输出的是每一个动作的概率?
        # 定义一个优化器,其实就是求梯度,就比如随机梯度下降,这里的Adam是一个优化器类型,相较于普通的SGD更快更稳定(就是wt=wt+....的功能，只不过实现的方法不一样)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # 更新学习率,学习率是不断更新的,不是一成不变的,这段代码就表示每5000步将学习率改变为原来的1/3
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        pi = nn.Softsign()(self.pi(x)) # [-1,1]

        return pi

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.chkpt_file), exist_ok=True)
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))