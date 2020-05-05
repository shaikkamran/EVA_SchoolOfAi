from __future__ import print_function



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import logging

logger=logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class conv_block(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=(3,3),dropout=0.1,**kwargs):
        super(conv_block, self).__init__()
        self.convblock = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,**kwargs),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            )
        
    def forward(self,x):
        return self.convblock(x)

class conv_block_without_relu(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=(3,3),dropout=0.1,**kwargs):
        super(conv_block_without_relu, self).__init__()
        self.convblock = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,**kwargs),
                nn.BatchNorm2d(out_channels),
                
                nn.Dropout(p=dropout)
            )
        
    def forward(self,x):
        return self.convblock(x)


class State():

  def __init__(self):
    
    self.image_list=[]
    self.orientation_list=[]

  def add(self,state_tuple):
    
    self.image_list.append(state_tuple.image)
    self.orientation_list.append(state_tuple.orientation)
    

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states,batch_next_states, batch_actions, batch_rewards, batch_dones = State(),State(), [], [], [] 
    
    for i in ind: 

      state,next_state, action, reward, done = self.storage[i]
      
      batch_states.add(state)
      batch_next_states.add(next_state)
      
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))

    return batch_states,batch_next_states,np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)
    



class Actor(nn.Module):
    def __init__(self,image_size,max_action,orientation_dim,out_channels):
        super(Actor, self).__init__()
        
        
        _,self.height,self.width=image_size
        self.max_action=max_action
        self.out_channels=out_channels
        self.orientation_dim=orientation_dim
        
        self.set_dimensions((self.height//2,self.width//2))
        
        self.conv1=conv_block(in_channels=3,out_channels=10,padding=(self.p_h,self.p_w),stride=2)
        
        self.set_dimensions((self.height//2,self.width//2))
        
        self.conv2 = conv_block(in_channels=10,out_channels=24,padding=(self.p_h,self.p_w),stride=2)
        
        self.set_dimensions((self.height,self.width))
        
        self.conv3=conv_block_without_relu(in_channels=24,out_channels=out_channels,padding=(self.p_h,self.p_w))
        
        
        self.gap=nn.AvgPool2d(self.height)
        self.linear1=nn.Linear(self.orientation_dim+self.out_channels,10)
        self.bn1 = nn.BatchNorm1d(num_features=10)
        
        self.linear2=nn.Linear(10,1)
        self.bn2 = nn.BatchNorm1d(num_features=1)
        
        
    def set_dimensions(self,output_dim):
        self.p_w,self.p_h=(1,1)
        self.height,self.width=output_dim
        
        
    def forward(self, x,u):

        x=self.conv3(self.conv2(self.conv1(x)))
        
        x=self.gap(x).view(-1,self.out_channels)
        
        logger.debug(f"The output of gap={x}")        
        # logger.debug(f"{x.shape} === {u.shape}")
        x=torch.cat([x,u],1)

        x=self.linear2(F.relu(self.bn1(self.linear1(x))))
        x=self.bn2(x)
        logger.debug(f"The output before tanh={x}")
        return self.max_action*torch.tanh(x)
    
class Critic(nn.Module):
    def __init__(self,image_size,action_dim,orientation_dim,critic_out_channels):
        super(Critic, self).__init__()
        
     
        
        _,self.height,self.width=image_size
        self.action_dim=action_dim
        self.out_channels=critic_out_channels
        self.orientation_dim=orientation_dim
        self.set_dimensions((self.height//2,self.width//2))
        
        self.conv1=conv_block(in_channels=3,out_channels=10,padding=(self.p_h,self.p_w),stride=2)
        
        self.set_dimensions((self.height//2,self.width//2))
        
        self.conv2=conv_block(in_channels=10,out_channels=24,padding=(self.p_h,self.p_w),stride=2)
        
        self.set_dimensions((self.height,self.width))
        
        self.conv3=conv_block_without_relu(in_channels=24,out_channels=self.out_channels,dropout=0.0,padding=(self.p_h,self.p_w))
        
        self.gap=nn.AvgPool2d(self.height)
        
        self.linear1=nn.Linear(self.out_channels + action_dim+orientation_dim, 10)
        
        self.bn1 = nn.BatchNorm1d(num_features=10)
        
        self.linear2=nn.Linear(10,1)
        self.bn2 = nn.BatchNorm1d(num_features=1)
        
        
        
        _,self.height,self.width=image_size
        self.action_dim=action_dim
        
        self.set_dimensions((self.height//2,self.width//2))
        
        self.conv1_2=conv_block(in_channels=3,out_channels=10,padding=(self.p_h,self.p_w),stride=2)
        
        self.set_dimensions((self.height//2,self.width//2))
        
        self.conv2_2=conv_block(in_channels=10,out_channels=24,padding=(self.p_h,self.p_w),stride=2)
        
        self.conv3_2=conv_block_without_relu(in_channels=24,out_channels=self.out_channels,dropout=0.0,padding=(self.p_h,self.p_w))
        
        self.gap_2=nn.AvgPool2d(self.height)
        
        self.linear1_2=nn.Linear(self.out_channels + action_dim+orientation_dim, 10)
        
        self.bn1_2 = nn.BatchNorm1d(num_features=10)

        self.linear2_2=nn.Linear(10,1)
        self.bn2_2 = nn.BatchNorm1d(num_features=1)
        
    def set_dimensions(self,output_dim):
        self.p_w,self.p_h=(1,1)
        self.height,self.width=output_dim
    
    def forward(self, x,orientation,action):
        
        x1=self.conv3(self.conv2(self.conv1(x)))
        x1=self.gap(x1).view(-1,self.out_channels)
        x1=torch.cat([x1,orientation,action],1)
        
        x1= self.bn2(F.relu(self.linear2(self.bn1(self.linear1(x1)))))
        
        
        
        x2=self.conv3_2(self.conv2_2(self.conv1_2(x)))
        x2=self.gap_2(x2)
        x2=x2.view(-1,self.out_channels)
        x2=torch.cat([x2,orientation,action],1)
        x2= self.bn2_2(F.relu(self.linear2_2(self.bn1_2(self.linear1_2(x2)))))
        
        return x1,x2
        
    def Q1(self,x,orientation,action):
        x1=self.conv3(self.conv2(self.conv1(x)))
        x1=self.gap(x1).view(-1,self.out_channels)
        x1=torch.cat([x1,orientation,action],1)
        x1= self.bn2(F.relu(self.linear2(self.bn1(self.linear1(x1)))))
        
        return x1   



class TD3(object):
  
  def __init__(self,image_size,action_dim, max_action,orientation_dim,out_channels,critic_out_channels):
    self.actor = Actor(image_size,max_action,orientation_dim,out_channels).to(device)
    self.actor_target = Actor(image_size,max_action,orientation_dim,out_channels).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(image_size,action_dim,orientation_dim,critic_out_channels).to(device)
    self.critic_target = Critic(image_size,action_dim,orientation_dim,critic_out_channels).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, image,orientation):
    # state = torch.Tensor(state.reshape(1, -1)).to(device)
    self.actor.eval()
    return self.actor(image,orientation).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states,batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      
      if len(batch_actions.shape)!=2:
        batch_actions=batch_actions.reshape(batch_size,1)

      
      state_image = torch.tensor(np.array(batch_states.image_list),dtype=torch.float).to(device)
      state_orientation=torch.tensor(np.array(batch_states.orientation_list),dtype=torch.float).to(device)

      next_state_image = torch.tensor(np.array(batch_next_states.image_list),dtype=torch.float).to(device)
      next_state_orientation=torch.tensor(np.array(batch_next_states.orientation_list),dtype=torch.float).to(device)
      
      
      action = torch.tensor(batch_actions,dtype=torch.float).to(device)
      reward = torch.tensor(batch_rewards,dtype=torch.float).to(device)
      done = torch.tensor(batch_dones,dtype=torch.float).to(device)
      
      # print(state.shape,next_state.shape,action.shape)
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state_image,next_state_orientation)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state_image, next_state_orientation,next_action)
      # logger.debug(f"Outputs of critic={target_Q1}  {target_Q2}")
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state_image, state_orientation,action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      logger.debug(f"critic loss is {critic_loss}")
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state_image, state_orientation,self.actor(state_image,state_orientation)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        logger.debug(f"Actor loss is {actor_loss}")
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
