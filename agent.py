import gymnasium
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
from experience_replay import ReplayMemory
from dqn import DQN
from datetime import datetime, timedelta
import argparse
import itertools
import flappy_bird_gymnasium
import os
from gymnasium.wrappers import RecordVideo
import imageio
import numpy as np
import time
import cv2

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force cpu, sometimes GPU not always faster than CPU due to overhead of moving data to GPU

class Agent:

    def __init__(self, hyperparameters_set):
        with open('/Users/yashrajkupekar/code/Reinforcement Learning/DQN-FlappyBird/hyperparameters.yml', 'r') as f:  # Correct file name
            all_hyperparameters_sets = yaml.safe_load(f)
            
        # Fetch the correct hyperparameters from the dictionary
        hyperparameters = all_hyperparameters_sets[hyperparameters_set]
        
        self.hyperparameter_set = hyperparameters_set
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']  # Fix key name (was 'epsilon min')
        self.env_id = hyperparameters['env_id']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate_a= hyperparameters ['learning_rate_a']# learning rate (alpha)
        self.discount_factor_g= hyperparameters['discount_factor_g']# discount rate (gamma)
        self.loss_fn = nn.MSELoss()
        self.optimizer = None   
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']

        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
    def run(self,is_training=True, render=False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = gymnasium.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        


        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
        
        reward_per_episode = []
        

        

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict()) # Initialize target network with same weights as policy network

            # Policy network optimizer. "Adam" optimizer can be swapped to something else.
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            epsilon_history = []
            step_count = 0

            best_reward = -99999999
        
        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch to evaluation mode
            policy_dqn.eval()
            
        for episode in itertools.count():
            state, _ = env.reset()

            #convert state to tensor
            state = torch.tensor(state,dtype=torch.float,device=device)
            terminated = False   

            episode_reward = 0.0

            while (not terminated and episode_reward < self.stop_on_reward):
                

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        #used argmax to get the action with highest Q-value and unsqueeze to add batch dimension
                        #then squeeze to remove batch dimension 
                        #because its needed for pytorch 
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                
                #convert new state to tensor
                new_state = torch.tensor(new_state,dtype=torch.float,device=device)

                #convert reward to tensor
                reward = torch.tensor(reward,dtype=torch.float,device=device)
                episode_reward += reward
                if is_training:
                    memory.append((state, action, reward, new_state, terminated))
                    
                    #Increment step count
                    step_count += 1
                    

                #move to the next state 
                state = new_state

            reward_per_episode.append(episode_reward) 



            # Save model when new best reward is obtained.
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(reward_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory)>self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0



    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def optimize(self, mini_batch, policy_dqn, target_dqn):
        
        # Directly extract components from the mini-batch using zip
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminations = zip(*mini_batch)
        
        # Stack tensors to create batch tensors
        states = torch.stack(batch_states)
        actions = torch.tensor(batch_actions, dtype=torch.long, device=device)
        rewards = torch.stack(batch_rewards)
        new_states = torch.stack(batch_next_states)
        terminations = torch.tensor(batch_terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            if self.enable_double_dqn:
                #first get the actions from policy network
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                
                #then get the Q values from target network
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).gather(dim=1,index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]


        current_q = policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()

        # Calculate loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update parameters



    def run_with_recording(self, is_training=False, record_video=True):
        """Run the agent and record video while showing gameplay in a window with score overlay"""
        
        # Set up video recording
        if record_video:
            video_folder = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_videos")
            os.makedirs(video_folder, exist_ok=True)
            video_path = os.path.join(video_folder, f"{self.hyperparameter_set}_episode_{int(time.time())}.mp4")
        
        # Create environment with rgb_array rendering
        env = gymnasium.make(self.env_id, render_mode='rgb_array', **self.env_make_params)
        
        # Load model for testing
        policy_dqn = DQN(env.observation_space.shape[0], env.action_space.n, 
                        self.fc1_nodes, self.enable_dueling_dqn).to(device)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
        policy_dqn.eval()
        
        # Set up video writer
        video_writer = None
        
        for episode in range(5):  # Run 5 episodes
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            
            episode_reward = 0.0
            score = 0  # Track the score (number of pipes passed)
            
            while not terminated and episode_reward < self.stop_on_reward:
                # Get action from policy
                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                
                # Take action in environment
                new_state, reward, terminated, _, info = env.step(action.item())
                
                # Update score (assuming reward of 1.0 is given when passing a pipe)
                if reward == 1.0:
                    score += 1
                
                # Render frame
                frame = env.render()
                
                # Convert RGB to BGR (OpenCV uses BGR)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Add score text to the frame
                cv2.putText(
                    frame_bgr, 
                    f"Score: {score}", 
                    (10, 30),  # Position (x, y) from top-left
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font
                    1,  # Font scale
                    (255, 255, 255),  # Color (white)
                    2,  # Thickness
                    cv2.LINE_AA  # Line type
                )
                
                # Add episode reward text
                cv2.putText(
                    frame_bgr,
                    f"Reward: {episode_reward:.1f}",
                    (10, 70),  # Position below score
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Display frame
                cv2.imshow('Flappy Bird', frame_bgr)
                cv2.waitKey(1)  # This is necessary to update the window
                
                # Initialize video writer if not already created
                if record_video and video_writer is None:
                    height, width, layers = frame_bgr.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
                    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                
                # Write frame to video
                if record_video:
                    video_writer.write(frame_bgr)
                
                # Convert to tensor for next iteration
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                episode_reward += reward
                
                # Move to next state
                state = new_state
            
            print(f"Episode {episode} finished with reward {episode_reward}, score: {score}")
        
        # Clean up
        if record_video and video_writer is not None:
            video_writer.release()
            print(f"Video saved to {video_path}")
        
        cv2.destroyAllWindows()
        env.close()

if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Hyperparameter set to use')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--record', help='Record video (only in test mode)', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameters_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        if args.record:
            dql.run_with_recording(is_training=False, record_video=True)
        else:
            dql.run(is_training=False, render=True)