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
import random
import shutil
seperator = ':'

ACTIONS = ['forward', 'backward', 'left', 'right']     # available actions
EPSILON = 0.1   # greedy police
ALPHA = 0.1   # learning rate(= 0.1)
GAMMA = 1    # discount factor(= 0.9)
LAMDA = 0.4
seperator = ':'

def build_q_table(Actions, S=None, q_table = None, first_call = None):
    if first_call:
        q_table = pd.DataFrame(columns=Actions)  
    else:
        if not S in q_table.index:
            q_table = q_table.append(pd.DataFrame(np.zeros((len([S]), len(Actions))), index = [S], columns=Actions))
    return q_table

def build_elig_table(Actions, S=None, elig_table = None, first_call = None):
    if first_call:
        elig_table = pd.DataFrame(columns=Actions)  
    else:
        if not S in elig_table.index:
            elig_table = elig_table.append(pd.DataFrame(np.zeros((len([S]), len(Actions))), index = [S], columns=Actions))
    return elig_table

def choose_action(state, q_table):
    # This is how to choose an action under our policy
    state_actions = q_table.loc[state, :]
    if (np.random.uniform(0.0, 1.0) < EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name

def move(S, A):
    if A == 'forward':
        agent_host.sendCommand("movesouth 1")
    elif A == 'backward':
        agent_host.sendCommand("movenorth 1")
    elif A == 'left':
        agent_host.sendCommand("moveeast 1")
    elif A == 'right':
        agent_host.sendCommand("movewest 1")
        
#def drawQ(q_table, curr_x=None, curr_y=None):
#    if canvas is None or root is None:
#        return
#    canvas.delete("all")
#    action_inset = 0.1
#    action_radius = 0.1
#    curr_radius = 0.2
#    action_positions = [ ( 0.5, action_inset ), ( 0.5, 1-action_inset ), ( action_inset, 0.5 ),  ( 1-action_inset, 0.5 )]
#    # (NSWE to match action order)
#    min_value = -20
#    max_value = 20
#    for x in range(world_x):
#        for y in range(world_y):
#            k = 0
#            S = "%d:%d" % (x,y)
#            canvas.create_rectangle( (world_x-1-x)*scale, (world_y-1-y)*scale, (world_x-1-x+1)*scale, (world_y-1-y+1)*scale, outline="#fff", fill="#000")
#            for action in ACTIONS:
#                if not S in q_table.index:
#                    continue
#                value = q_table.loc[S, action]
#                color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
#                color = max( min( color, 255 ), 0 ) # ensure within [0,255]
#                color_string = '#%02x%02x%02x' % (255-color, color, 0)
#                canvas.create_oval( (world_x - 1 - x + action_positions[k][0] - action_radius ) *scale,
#                                         (world_y - 1 - y + action_positions[k][1] - action_radius ) *scale,
#                                         (world_x - 1 - x + action_positions[k][0] + action_radius ) *scale,
#                                         (world_y - 1 - y + action_positions[k][1] + action_radius ) *scale, 
#                                         outline=color_string, fill=color_string )
#                k+=1
#    if curr_x is not None and curr_y is not None:
#        canvas.create_oval( (world_x - 1 - curr_x + 0.5 - curr_radius ) * scale, 
#                                 (world_y - 1 - curr_y + 0.5 - curr_radius ) * scale, 
#                                 (world_x - 1 - curr_x + 0.5 + curr_radius ) * scale, 
#                                 (world_y - 1 - curr_y + 0.5 + curr_radius ) * scale, 
#                                 outline="#fff", fill="#fff" )
#    root.update()
    
for k in range(150): #loop for training runs  

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
                                <DrawBlock x="4"  y="45" z="10" type="diamond_block" />
                              </DrawingDecorator>
                              <ServerQuitFromTimeUp timeLimitMs="20000"/>
                              <ServerQuitWhenAnyAgentFinishes/>
                            </ServerHandlers>
                          </ServerSection>
                        
                          <AgentSection mode="Survival">
                            <Name>MalmoTutorialBot</Name>
                            <AgentStart>
                              <Placement x="4.5" y="46.0" z="1.5" pitch="30" yaw="0"/>
                            </AgentStart>
                            <AgentHandlers>
                              <DiscreteMovementCommands/>
                              <ObservationFromFullStats/>
                                  <ObservationFromGrid>
                                      <Grid name="floor3x3">
                                          <min x="-1" y="-1" z="-1"/>
                                          <max x="1" y="-1" z="1"/>
                                      </Grid>
                                  </ObservationFromGrid>
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
    
    my_mission = MalmoPython.MissionSpec(missionXML, True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo( 256, 144  ) #16:9 aspect ratio
    my_mission.setViewpoint( 0 ) #front person view
    
    
    #-------------------Code to use a second or different client --------------------
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available
    agentID = 0
    expID = 'tabular_backward_SARSA'
    #agent_host.startMission( my_mission, my_clients, my_mission_record, agentID, "%s-%d" % (expID, i)) #replace to start on a different client
    
    q_table = build_q_table(ACTIONS, first_call=True)
    elig_table = build_elig_table(ACTIONS, first_call=True)
    
#    # -- set up the python-side drawing -- #
#    scale = 40
#    world_x = 6
#    world_y = 12
#    root = tk.Tk()
#    root.wm_title("Q-table_Backward_SARSA")
#    canvas = tk.Canvas(root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
#    canvas.grid()
#    root.update()
    
    #Setup the csv file which stores Episode and Reward for the Episode
    path = "/home/skripsie/Work/CSVFiles/CliffWalking_backward_SARSA_grid_%d.csv" % k
    file = open(path, 'w')
    thewriter = csv.writer(file)
    thewriter.writerow(['Reward'])
    
    for i in range(750): #loop for episodes
        
        if i > 250:
            my_mission.drawBlock( 3,45,3,"lava")
            my_mission.drawBlock( 2,45,4,"lava")
            my_mission.drawBlock( 1,45,5,"lava")
            my_mission.drawBlock( 0,45,5,"lava")
            my_mission.drawCuboid(5,45,9,2,45,9,"lava")
        if i > 500:
            EPSILON = EPSILON/1.01
        if i == 700:
            EPSILON = 0
            
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
        print("Mission running ", end=' ')
        
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
    
        #Initialise S
        S = '-1:-1:-1:0:0:-1:0:-1:-1'
     
        q_table = build_q_table(ACTIONS, S, q_table)
        elig_table = build_elig_table(ACTIONS, S, elig_table)
        
        R_total = 0
    
        A = choose_action(S, q_table) #follow your policy
      
        # Loop until mission ends:
        while world_state.is_mission_running: 
            
            move(S, A) #make your move
            
            time.sleep(0.2)
            world_state = agent_host.getWorldState() #see how the environment changed given our move
            if world_state.is_mission_running and world_state.number_of_observations_since_last_state > 0 and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                obs_text = world_state.observations[-1].text
                obs = json.loads(obs_text)
                grid = obs.get(u'floor3x3', 0)
                grid = [0 if (x=='sandstone' or x=='cobblestone') else x for x in grid]
                grid = [-1 if (x=='flowing_lava' or x=='lava') else x for x in grid]
                grid = [1 if x=='diamond_block' else x for x in grid]
                S_ = '%s' %(seperator.join(map(str, grid)))
                q_table = build_q_table(ACTIONS, S_, q_table)
                elig_table = build_elig_table(ACTIONS, S_, elig_table)
                A_ = choose_action(S_, q_table)
            
            R = sum(r.getValue() for r in world_state.rewards) #what reward did we get?
            R_total += R #for summing the total reward of the run
            
            elig_table.loc[S, A] += 1
            elig_table.loc[:,:] = GAMMA * LAMDA * elig_table.loc[:,:]
            q_current = q_table.loc[S, A]
            if world_state.is_mission_running: #update Q value
                q_next = q_table.loc[S_, A_]
                delta = R + GAMMA * q_next - q_current
            if not world_state.is_mission_running: #update Q value from terminal state
                delta = R - q_current
            q_table.loc[:, :] += (ALPHA * delta * elig_table.loc[:, :])
            
            if world_state.is_mission_running:
                A = A_
                S = S_    
#            drawQ(q_table, curr_x = int(obs[u'XPos']), curr_y = int(obs[u'ZPos']))
        
        print(R_total)
#        drawQ(q_table)  
        
        #write into the csv file
        thewriter = csv.writer(file)
        thewriter.writerow([R_total]) 
        
        print()
        print("Mission ended, episode = %d" % (i+1))
        print()
        # Mission has ended.
    file.close()
    print('Done')
#    root.destroy()
    time.sleep(5)
    
#    #------------- Option to keep Q-table open -----------------------------------
#    while True:
#        time.sleep(5)
