from __future__ import print_function


from future import standard_library
standard_library.install_aliases()
from builtins import range
import MalmoPython
import json
import sys
import time
import numpy as np
import pandas as pd
import tkinter as tk
import csv
import shutil

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 8, kernel_size=3, stride=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w,3,1),2,2),3,1),2,2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h,3,1),2,2),3,1),2,2)
        linear_input_size = convw * convh * 8
        print(linear_input_size)
        self.fc1 = torch.nn.Linear(linear_input_size, 64)
        self.head = nn.Linear(64, outputs)
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.head(x)
        return x


    
def get_screen(world_state):
    frame = world_state.video_frames[0] 
    image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) )
    image = np.asarray(image)
    screen = image.transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    resize = T.Compose([T.ToPILImage(),
                    T.Resize((40, 40)),
                    T.ToTensor()])
    return resize(screen).unsqueeze(0)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > 0:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
        return action


def move(A):
    #moves given Action A
    if A == 0: #forward
        agent_host.sendCommand("movesouth 1")
    elif A == 1: #backward
        agent_host.sendCommand("move 1")
    elif A == 2: #left
        agent_host.sendCommand("moveeast 1")
    elif A == 3: #right
        agent_host.sendCommand("movewest 1")
        

        

for k in range(30): #loop for training runs 

    #------------------- Building the Mission --------------------
    missionXML= '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                        
                          <About>
                            <Summary>Cliff walking mission based on Sutton and Barto.</Summary>
                          </About>
                        
                          <ServerSection>
                            <ServerInitialConditions>
                                <Time><StartTime>1</StartTime></Time>
                            </ServerInitialConditions>
                            <ServerHandlers>
                              <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1"/>
                              <DrawingDecorator>
                                <DrawCuboid x1="-5" y1="46" z1="-2" x2="7" y2="50" z2="13" type="air" />            
                                <DrawCuboid x1="-5" y1="45" z1="-2" x2="7" y2="45" z2="13" type="lava" />          
                                <DrawCuboid x1="-2"  y1="45" z1="1"  x2="3" y2="45" z2="10" type="sandstone" />      
                                <DrawBlock x="4"  y="45" z="1" type="cobblestone" />    
                                <DrawBlock x="-2"  y="45" z="10" type="diamond_block" />
                              </DrawingDecorator>
                              <ServerQuitFromTimeUp timeLimitMs="20000"/>
                              <ServerQuitWhenAnyAgentFinishes/>
                            </ServerHandlers>
                          </ServerSection>
                        
                          <AgentSection mode="Survival">
                            <Name>MalmoTutorialBot</Name>
                            <AgentStart>
                              <Placement x="4.5" y="46.0" z="1.5" pitch="45" yaw="0"/>
                            </AgentStart>
                            <AgentHandlers>
                              <DiscreteMovementCommands/>
                              <ObservationFromFullStats/>
                                  <VideoProducer viewpoint="0" want_depth="false">
                                    <Width>320</Width>
                                    <Height>240</Height>
                                  </VideoProducer>
                              <RewardForTouchingBlockType>
                                <Block reward="-100.0" type="lava" behaviour="onceOnly"/>
                                <Block reward="100.0" type="diamond_block" behaviour="onceOnly"/>
                              </RewardForTouchingBlockType>
                              <RewardForSendingCommand reward="-1" />
                              <AgentQuitFromTouchingBlockType>
                                  <Block type="lava" />
                                  <Block type="diamond_block" />
                              </AgentQuitFromTouchingBlockType>
                            </AgentHandlers>
                          </AgentSection>
                        
                        </Mission>'''
            
    # Create default Malmo objects:

    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)
        
    agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
    agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
    my_mission = MalmoPython.MissionSpec(missionXML, True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo( 256, 144   ) #16:9 aspect ratio
    my_mission.setViewpoint( 0 ) #front person view
    
    #------------- Code to use a second or different client
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available
    agentID = 0
    expID = 'deep_q_learning'
    #agent_host.startMission( my_mission, my_clients, my_mission_record, agentID, "%s-%d" % (expID, i)) #replace to start on a different client
    #------------------- END Building the Mission --------------------
    
    #Setup the csv file which stores Episode and Reward for the Episode
    path = "/home/skripsie/Work/CSVFiles/CliffWalking_DQN_random_trained_0.%d.csv" % k
    file = open(path, 'w')
    thewriter = csv.writer(file)
    thewriter.writerow(['Reward'])
    
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.1
    EPS_DECAY = 45
    TARGET_UPDATE = 10
    n_actions = 4
    lr = 0.001
    
    
    screen_height = 40 
    screen_width = 40

    policy_net = DQN(screen_height, screen_width, n_actions)
    target_net = DQN(screen_height, screen_width, n_actions)
    
    policy_net = torch.load("NN_new_CNN_small_prob")
    policy_net.eval()
    target_net = torch.load("NN_new_CNN_small_prob")
    target_net.eval()

#    target_net.load_state_dict(policy_net.state_dict())
#    target_net.eval()
#    
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
    memory = ReplayMemory(1000)
    
    
    steps_done = 0
    
    
    for i in range(750): #loop for episodes
        if i == BATCH_SIZE:
            steps_done = 0
        
        xrange = [3,2,1,0,-1,-2]
        zrange = [9,8,7,6,5,4,3,2]
        x1 = int(np.random.choice(xrange))
        z1 = int(np.random.choice(zrange))
        
        x2 = int(np.random.choice(xrange))
        z2 = int(np.random.choice(zrange))
        
        x3 = int(np.random.choice(xrange))
        z3 = int(np.random.choice(zrange))
        
        x4 = int(np.random.choice(xrange))
        z4 = int(np.random.choice(zrange))
        
        x5 = int(np.random.choice(xrange))
        z5 = int(np.random.choice(zrange))
        
        x6 = int(np.random.choice(xrange))
        z6 = int(np.random.choice(zrange))
        
        x7 = int(np.random.choice(xrange))
        z7 = int(np.random.choice(zrange))
        
        my_mission.drawBlock( x1,45,z1,"lava")
        my_mission.drawBlock( x2,45,z2,"lava")
        my_mission.drawBlock( x3,45,z3,"lava")
        my_mission.drawBlock( x4,45,z4,"lava")
        my_mission.drawBlock( x5,45,z5,"lava")
        my_mission.drawBlock( x6,45,z6,"lava")
        my_mission.drawBlock( x7,45,z7,"lava")
        
    #------------------- Starting the Mission --------------------    
        # Attempt to start a mission:
        max_retries = 3
        for retry in range(max_retries):
            try:
    #            agent_host.startMission( my_mission, my_mission_record )
                agent_host.startMission( my_mission, my_clients, my_mission_record, agentID, "%s-%d" % (expID, i)) #replace to start on a different client
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2)
        
        # Loop until mission starts:
        print("Waiting for the mission to start ", end=' ')
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)
        
        print()
        print('Mission Started')
    #------------------- END Starting the Mission --------------------
     
    #------------------- Determine the initial S --------------------    
        # wait for a valid observation
        world_state = agent_host.peekWorldState()
        while world_state.is_mission_running and all(e.text=='{}' for e in world_state.observations):
            world_state = agent_host.peekWorldState()
        # wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = agent_host.peekWorldState()
        world_state = agent_host.getWorldState()
        for err in world_state.errors:
            print(err)
        
#        last_screen = get_screen(world_state)
#        current_screen = get_screen(world_state)
#        state = current_screen - last_screen
        state = get_screen(world_state)
        
    #------------------- Can be used to see what the image looks like -------------------- 
#        _, _, screen_height, screen_width = state.shape
#        print(screen_height, screen_width)
#        
#        plt.figure()
#        plt.imshow(state.cpu().squeeze(0).permute(1, 2, 0).numpy(),
#                   interpolation='none')
#        plt.title('Example extracted screen')
#        plt.show()
        

        print(policy_net(state))
    #------------------- END Determine the initial S --------------------  
        R_total = 0 #Sum of Rewards
        framecount=0

      
        while world_state.is_mission_running:
            
            
            action = select_action(state)
            move(action.item())
            time.sleep(0.2)
#            plt.figure()
#            plt.imshow(state.cpu().squeeze(0).permute(1, 2, 0).numpy(),
#                       interpolation='none')
#            plt.title('Example extracted screen')
#            plt.show()

            world_state = agent_host.peekWorldState()
            while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
                world_state = agent_host.peekWorldState()                
            world_state = agent_host.getWorldState()

            if world_state.is_mission_running:
                next_state = get_screen(world_state)
#                    last_screen = current_screen
#                    current_screen = get_screen(world_state)
#                    next_state = current_screen - last_screen
                    
                
#                framecount +=1
#                frames = world_state.video_frames
#                frame = frames[0]
#                image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels))
#                image.save("%d.png" % framecount)
#                    
                    
#                plt.figure()
#                plt.imshow(next_state.cpu().squeeze(0).permute(1, 2, 0).numpy(),
#                          interpolation='none')
#                plt.title('Example extracted screen')
#                plt.show()
                    
#                    print("oye")
                
            elif not world_state.is_mission_running:
                next_state = None
                
            reward = sum(r.getValue() for r in world_state.rewards)
            
            
            R_total += reward
            reward = torch.tensor([reward], dtype=torch.float)
       
            memory.push(state, action, next_state, reward)
            
            # Move to the next state
            state = next_state

 
        optimize_model()
            
        # Update the target network, copying all weights and biases in DQN
        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(target_net, "NN_random_with_testing_onSmall")
        steps_done += 1  
        print()        
        print(R_total)  
        
        #write into the csv file
        thewriter = csv.writer(file)
        thewriter.writerow([R_total]) 
        
        my_mission.drawBlock( x1,45,z1,"sandstone")
        my_mission.drawBlock( x2,45,z2,"sandstone")
        my_mission.drawBlock( x3,45,z3,"sandstone")
        my_mission.drawBlock( x4,45,z4,"sandstone")
        my_mission.drawBlock( x5,45,z5,"sandstone")
        my_mission.drawBlock( x6,45,z6,"sandstone")
        my_mission.drawBlock( x7,45,z7,"sandstone")
        
        
        print()
        print("Mission ended, episode = %d" % (i+1))
        print()
        # Mission has ended.
    
    file.close()
    print('Done')

    

