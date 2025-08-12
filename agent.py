import torch as T
from networks import ActorNetwork, CriticNetwork
import numpy as np

class Agent:
    # actor_dims, critic_dims 两个网络输入层的输入维度
    # n_actions动作维度
    # n_agents智能体数量（用于critic）
    # agent_idx当前智能体编号（用于命名）
    # chkpt_dir模型保存路径
    # alpha, beta actor和critic的学习率
    # fc1, fc2全连接隐藏层大小
    # gamma折扣因子
    # tau软更新参数（用于目标网络更新）
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.0001, beta=0.0002, fc1=128,
                    fc2=128, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx   #这里如果agent_idx=0,那么self.agent_name 的值就为 agent_0,%s只是一个占位符
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        # tau=1 是为了初始化时让目标网络和主网络完全一致，之后训练时用 tau=0.01 进行缓慢的软更新，以提高训练稳定性。
        # 具体的内容是在下面的函数中实现的
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, time_step, evaluate=False):
        # 如果调用者没有传入 evaluate 参数，就默认使用 False；
        # 如果调用者明确传入了 evaluate=True，那么这个函数执行时 evaluate 的值就是 True。
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)#感觉是一个二维的（最起码）---使用阶段是一维的，训练阶段是二维的，训练是全局环境
        # 前向传播就是利用现有的网络参数和神经网路框架算出action的值？
        # 其实就是Π（a|s,参数），直接输出的就是当前参数下，对应环境下的最优动作，这个动作网络就是这个策略Π的函数形式，只不过在Maddpg优化的时候是用到目标函数的，那个也会用到这个策略Π，而且也会优化参数，
        # 类似的critic网络也是代表q(a|s,参数)，直接输出的就是当前参数下，对应环境下的action value，也是在优化的时候，才涉及目标函数，本质上都是将他们用函数表示，
        # 当然在优化的时候，肯定还会用到，但是并代表输出的结果就是优化的目标
        actions = self.actor.forward(state)

        # exploration
        max_noise = 0.75
        min_noise = 0.01
        decay_rate = 0.999995

        noise_scale = max(min_noise, max_noise * (decay_rate ** time_step))
        #self.n_actions表示维度,是个向量的形式
        noise = 2 * T.rand(self.n_actions).to(self.actor.device) - 1 # [-1,1)#这是是基础噪声向量（类似于单位向量） ---后续是*噪声强度是真实的噪声
        # 训练阶段添加探索
        if not evaluate:
            noise = noise_scale * noise
        else:
            noise = 0 * noise
        
        action = actions + noise
        # .numpy() 不能在 GPU 上操作；
        action_np = action.detach().cpu().numpy()[0]
        # 裁剪---原因减小每一步移动的幅度,避免训练的不稳定,加速收敛1
        # 要求训练阶段和测试阶段相同,要不然在训练阶段,利用剪切能够较好的实现功能,但是如果测试的时候不再进行剪切,那么无人机的运动形式可能就和训练阶段不一样,就会导致模型不能适用
        magnitude = np.linalg.norm(action_np)
        if magnitude > 0.04:
            action_np = action_np / magnitude * 0.04
        #     返回的是每一个动作的概率，不是给出确定的动作
        return action_np

    # 更新网络的参数(目标网络)
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # named_parameters()是PyTorchnn.Module提供的函数，可以返回模型中所有需要训练的参数及其名称，即使你没有在自己写的类中定义它，只要继承了
        # nn.Module就能直接使用。它只返回参与梯度计算（也就是可训练的）参数，不会包含那些 requires_grad=False 的参数。
        # 网络在进行初始化的时候，会默认初始化参数，这些参数是保存在内存中，不用每一次都保存在文件当中
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()
        # 模型参数列表 转成 字典，这样后面可以用 名字（字符串）直接访问对应的参数张量。---转换的主要原因是方便后续的操作---但是操作的是副本
        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        # clone()的作用:为什么要用clone()?
        # .clone() 的作用是复制张量数据，避免引用混乱或内存共享问题，在参数更新时必须使用它来确保计算安全、稳定，避免梯度追踪错误或 in-place 覆盖风险。
        # 如果不用clone,会产生“in-place 操作”或内存共享问题。
        # a = x  # 实际上a和x指向同一块内存
        # a = a + y  # 现在你以为a是新值，其实x也可能被影响了！
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        # target_actor_params 是真实引用，修改它会立即改变模型参数；
        # target_actor_state_dict 是参数的副本，修改它不会影响模型，除非手动 load_state_dict()。-*----改变的是内存中的变量值
        # named_parameters 直通车，改了就动模型他；
        # state_dict 是快照，改了还得再装下。

        # 目标网络每次软更新后不需要保存，只有在你真正需要“记录当前最优模型”或“训练结束”时再调用
        # save_models()保存一次即可。
        # 不需要每次软更新都保存到文件中！
        # 在训练过程中，内存中的模型参数会一直保留并更新，
        # 只需要在训练结束或达到特定条件（如验证最优）时统一保存即可。

        # 在你定义网络层（如 nn.Linear）的同时，PyTorch 已经自动创建好权重和偏置等参数。
        # 你虽然看不见它们在代码中显式写出，但它们确实存在，而且可以通过 state_dict()、named_parameters() 等方式访问、查看、修改。

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
