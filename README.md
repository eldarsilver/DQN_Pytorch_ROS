# DQN_Pytorch_ROS

The goal of this project is to train Reinforcement Learning algorithms using Pytorch in a simulated environment rendered by OpenAI Gym and Gazebo, controlling the agent through ROS (Robot Operating System). Finally, the trained models will be deployed in a real world scenario using the robot called Turtlebot. 

## ENVIRONMENT, AGENT, TASK, ACTIONS AND REWARDS

The simulated environment generated by OpenAI Gym and Gazebo is a Maze and the agent is a virtual Turtlebot which has a mobile platform with a lidar on its top.

The task to solve is to learn a policy so the robot isn't closer than a configurable distance (0.2 meters by default) from any obstacle in the directions of the laser beams captured and filtered in each state. This value can be changed in the file `$HOME/python3_ws/src/openai_ros/openai_ros/src/openai_ros/task_envs/turtlebot2/config/turtlebot2_maze.yaml` through the variable:
```
min_range: 0.2 # Minimum meters below wich we consider we have crashed
```

The actions will be: move forward (`action = 0`), turn left (`action = 1`) and turn right (`action = 2`). These discretized actions will be translated in physical movements in the robot. These settings can be found in the same configuration file called `$HOME/python3_ws/src/openai_ros/openai_ros/src/openai_ros/task_envs/turtlebot2/config/turtlebot2_maze.yaml`. An example of these values is:
```
linear_forward_speed: 0.5 # Spawned for ging fowards
linear_turn_speed: 0.1 # Lienare speed when turning
angular_speed: 0.8 # 0.3 Angular speed when turning Left or Right
init_linear_forward_speed: 0.0 # Initial linear speed in which we start each episode
init_linear_turn_speed: 0.0 # Initial angular speed in shich we start each episode
```

The rewards can also be found in the same file `turtlebot2_maze.yaml`:
```
forwards_reward: 0 # Points Given to go forwards (5 points by default but changed to 0)
turn_reward: 0 # Points Given to turn as action (4 points by default but changed to 0)
end_episode_points: 2 # Points given when ending an episode (A '-' sign will be placed to this amount in the code ,i.e. -2. 200 by default but changed to 2)
```

A thing that needs to be clarified is that the episode ends when some of the laser readings contain a value less than `min_range` and when this happens, then the reward of that step will be `- end_episode_points` (i.e. -2 points).

## HOW TO CAPTURE EACH STATE

In this project, a RPLidar A1 has been used to capture each state of the environment. It emits 360 laser beams with 1º between 2 contiguous scans. 

Each reading will be stored in a Python list of 360 positions, so in the index 0 will be the laser beam correspondig to 0º or the robot's back side, and it follows a counter clockwise fashion. The index 89 will have the laser beam corresponding to the right side of the robot, the index 179 will contain the laser beam of the front of the robot and so on.

The settings of this physical Lidar will be fixed in the simulated Gym Gazebo environment (see the Gazebo 9 Installation section).

## INSTALLATION USING DOCKER (EASY WAY)

The main software components needed in this project are ROS Kinetic (used to manage the robot), OpenAI Gym with Gazebo (whose purpose is to simulate and visualize the Maze environment and the virtual robot agent with its sensors) and Pytorch to develop Neural Networks. 

As each version of ROS is dependent on a specific Ubuntu distro (i.e. ROS Kinetic needs Ubuntu 16.04), and ROS works with Python 2.7 by default but we 're going to use Pytorch with Python 3, the manual installation of the software requirements directly on the host OS is tedious with many potential conflicts due to Python versions.

Because of that, A Dockerfile is offered to simplify the installation process. But if you have enough patient and time available, you'll find the steps to follow to perform the installation manually.

Next, you'll see how to get a functional environment with ROS Kinetic, OpenaAI Gym, Gazebo, Pytorch and Python 3.

### Clone this repository

The first step will be to create a workspace or folder called `python3_ws` for the entire project:
```
   mkdir -p $HOME/python3_ws
   cd $HOME/python3_ws/
   mkdir -p src
   cd src
```
Then, you'll clone this repository:
```
   git clone <repository_url>
```

It will download a folder called `DQN_Pytorch_ROS` whose content (folders and files) has to be copied under the `python3_ws/src` folder. After that, the folder `DQN_Pytorch_ROS` has to be removed using:
```
sudo rm -R $HOME/python3_ws/src/DQN_Pytorch_ROS/
```

Next, you will have to download the folder called `kinetic-gazebo9` from this Google Drive url and place it under `python3_ws/src/` folder:

[Link to kinetic-gazebo9 folder](https://drive.google.com/drive/folders/1nt_306F5p9gr2eFns0IJDbVZbJcwrAYZ?usp=sharing)

The final folder structure should be:
``` 
   python3_ws/
      src/
        kinetic-gazebo9/
        laser_values/
        openai_ros/
        turtle2_openai_ros_example/
        xacro/
	Dockerfile
	LICENSE
	README.md
        requirements.txt
```

### Build Dockerfile

Once you are placed at `python3_ws/src`, a Docker image will be built from the provided `Dockerfile` using the following command:
```
docker build . -t <name_of_your_image>
```

### Run Docker Image

The built Docker Image will contain Ubuntu 16.04, ROS Kinetic, a modified version of Gazebo 9, Deep Learning libraries like Pytorch, OpenAI Gym or Tensorboard, other required packages and the `src` folder of this repo. It exposes the PORT 6006 for Tensorboard and 5900 like the PORT of a VNC Server.

As Gazebo needs to open a GUI window to show the Maze environment, we'll need to allow the root user of the Docker image to access the running X server of your host Linux OS:
```
xhost +local:root
```

After that command, you can launch the Docker image running:
```
docker run -it --privileged --rm -e DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v $HOME/python3_ws/src/turtle2_openai_ros_example/src/logs:/python3_ws/src/turtle2_openai_ros_example/src/logs -v $HOME/python3_ws/src/turtle2_openai_ros_example/src/checkpoints:/python3_ws/src/turtle2_openai_ros_example/src/checkpoints -v $HOME/python3_ws/src/turtle2_openai_ros_example/src/trace:/python3_ws/src/turtle2_openai_ros_example/src/trace -p 5900:5900 -p 6006:6006 <name_of_your_image>
```

Four volumes have been mounted between the host OS and the Docker container:
* A volume to share the Unix domain socket file (.X11-unix) of the host X server with the Docker container. 
* A volume to store the logs for Tensorboard events files
* A volume to save the Pytorch Policy Network checkpoints
* A volume to store JSON files of traces with information about the states and actions taken in each step of each episode of the training process.

Besides that, there are 2 ports mapped between the host OS and the Docker container:
* The port 5900 on which a VNC Server will be listening to.
* The port 6006 on which Tensorboard will be listening to.

When the `docker run` has been executed, a prompt will be shown. You will be inside the Docker container using the root account. 

You can jump to the `TRAIN DQN TO SOLVE MAZE ENVIRONMENT` section to kown more about this process but as a brief summary, you could train a virtual Turtlebot agent to learn a policy to solve the task in the Maze environment launching this command inside the Docker container:
```
roslaunch turtle2_openai_ros_example start_training_maze_v2_dqn.launch
```

You will see a lot of traces in the Docker container Terminal showing the value of each state, the action taken, the epsilon value at each step for the epsilon-greedy decision making, cumulated rewards, etc. All this information will be summarized and stored in contiguous JSON files in the `trace` folder. Each JSON file will contain the next 1000 transitions so they can be opened without memory issues.

The way to launch Tensorboard and inspect the rewards and the parameters of the Policy Network during the training phase would be to open a new Terminal in the host OS where Tensorboard should be installed and execute (Tensorboard is also installed in the Docker image so you could launch it from the Docker container):
```
tensorboard  --logdir=$HOME/python3_ws/src/turtle2_openai_ros_example/src/logs
```

When the training process has finished, you can make sure that the Gazebo server has been closed properly, looking for its pid (you have to search the `gzserver --verb` entry after calling ps aux command) and killing it. You have to do that because closing the Gazebo GUI window doesn't kill the Gazebo server process:
```
ps aux
kill -9 <PID of gzserver>
```

You can find the details to test the trained DQN model visualizing the results using Gazebo in the `TEST THE TRAINED MODEL IN THE OPENAI GYM AND GAZEBO ENVIRONMENT` but you can achive that running this command inside the Docker container after training the model or you can use the `dqn-final-episode-2671-step-110007.pt` file provided and placing it in the folder `/python3_ws/src/turtle2_openai_ros_example/src/checkpoints/` of the Docker container:
```
roslaunch turtle2_openai_ros_example start_test_maze_v2_dqn.launch
```

As a security measure, when you have finished using the Docker container, this command should be executed:
```
xhost -local:root
```

## MANUAL INSTALLATION ON HOST OS (HARD WAY)

This project needs to be installed and executed in Ubuntu 16.04 because ROS Kinetic will be used to control the robot.

###  Workspace folder and Virtual Environment

You should create a directory to store the workspace for the entire project:
```
   mkdir -p python3_ws
   cd python3_ws/
   mkdir -p src
```

Next, a virtual environment will be created (using virtualenv, conda, etc) to install all the requiered software.
```
   virtualenv py3envros --python=python3
   source py3envros/bin/activate
   sudo apt-get install python3-tk
   cd src/
```

### ROS Kinetic Installation

The following commands will be executed to install ROS Kinetic inside the virtual environment:
```
   sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
   sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
   sudo apt-get update
   sudo apt-get install ros-kinetic-desktop-full
   sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
   sudo rosdep init
   sudo apt-get install rospkg
   source /opt/ros/kinetic/setup.bash     
```

### Clone this repository 

The repository should be cloned under the folder `python3_ws/src/`:
```
   git clone <repository_url>
```
It will download a folder called `DQN_Pytorch_ROS` whose content (folders and files) has to be copied under the `python3_ws/src` folder. After that, the folder `DQN_Pytorch_ROS` has to be removed using:
```
sudo rm -R $HOME/python3_ws/src/DQN_Pytorch_ROS/
```

The folder structure should be:
``` 
   python3_ws/
      py3envros/
      src/
        laser_values/
        openai_ros/
        turtle2_openai_ros_example/
        xacro/
	Dockerfile
	LICENSE
	README.md
        requirements.txt
```
The configuration file `$HOME/python3_ws/src/turtle2_openai_ros_example/config/turtlebot2_openai_qlearn_params_v2.yaml` should be modified to indicate the correct value for the variable that stores the path to the ROS workspace (`python_ws` folder):
```
ros_ws_abspath: "/<correct_path>/python3_ws"
```

The `requirements.txt` will be installed:
```
   pip install -r requirements.txt
```

At this time, the folders called `xacro` and `openai_ros` will contain the right configuration data. But if you want to perform a fresh installation of these packages you should follow the next steps to achive the current state: 

(Do it only if you want to install a fresh version of `openai_ros` and `xacro` and configure them inside the folder `python3_ws/src/`)
```
   git clone https://bitbucket.org/theconstructcore/openai_ros.git
   cd openai_ros/
   git checkout version2
   cd ..
   git clone -b kinetic-devel https://github.com/ros/xacro.git
```
The file called `xacro/xacro.py` should be modified to properly import the functions coded in the remaining scripts located in `python3_ws/src/xacro/src/xacro`: `cli.py`, `color.py`, `initxacro.py` and `xmlutils.py` so xacro.py could call the functions of these scripts.

Regarding the `openai_ros` folder, the following files should be modified:

The configuration file `python3_ws/src/openai_ros/openai_ros/src/openai_ros/task_envs/turtlebot2/config/turtlebot2_maze.yaml` should have these values:
```
   n_actions: 3 # We have 3 actions, Forwards,TurnLeft,TurnRight
   n_observations: 5 # We have 5 different observations
   speed_step: 0.5 # Time to wait in the reset phases
   linear_forward_speed: 0.5 # Spawned for going fowards
   linear_turn_speed: 0.1 # Linear speed when turning
   angular_speed: 0.8 # 0.3 Angular speed when turning Left or Right
   init_linear_forward_speed: 0.0 # Initial linear speed in which we start each episode
   init_linear_turn_speed: 0.0 # Initial angular speed in shich we start each episode 
   new_ranges: 5 # How many laser readings we jump in each observation reading, the bigger the less laser resolution
   min_range: 0.2 # Minimum meters below wich we consider we have crashed
   max_laser_value: 6 # Value considered Ok, no wall
   min_laser_value: 0 # Value considered there is an obstacle or crashed
   forwards_reward: 5 # Points Given to go forwards
   turn_reward: 4 # Points Given to turn as action
   end_episode_points: 200 # 200 Points given when ending an episode
```
The function `def discretize_observation()` of the script `python3_ws/src/openai_ros/openai_ros/src/openai_ros/task_envs/turtlebot2/turtlebot2_maze.py` should look like this:
```
   def discretize_observation(self, data, new_ranges):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False
        discretized_ranges = []
        filtered_range = []
        mod = new_ranges
        max_laser_value = data.range_max
        min_laser_value = data.range_min
        rospy.logdebug("data = " + str(data))
        rospy.logwarn("len(data.ranges) = " + str(len(data.ranges)))
        rospy.logwarn("mod=" + str(mod))
        idx_ranges = [89, 135, 179, 224, 269]
        for item in idx_ranges:
            if data.ranges[item] == float('Inf') or numpy.isinf(data.ranges[item]):
                discretized_ranges.append(round(max_laser_value, self.dec_obs))
            elif numpy.isnan(data.ranges[item]):
                discretized_ranges.append(round(min_laser_value, self.dec_obs))
            else:
                discretized_ranges.append(round(data.ranges[item], self.dec_obs))
            if (self.min_range > data.ranges[item] > 0):
                rospy.logerr("done Validation >>> data.ranges[" + str(item) + "]=" + str(data.ranges[item])+"< "+str(self.min_range))
                self._episode_done = True
            else:
                rospy.logwarn("NOT done Validation >>> data.ranges[" + str(item) + "]=" + str(data.ranges[item])+"< "+str(self.min_range))
        rospy.logdebug("Size of observations, discretized_ranges==>"+str(len(discretized_ranges)))
        self.publish_filtered_laser_scan(laser_original_data=data, new_filtered_laser_range=discretized_ranges)
        return discretized_ranges
```
The most important part of this code is the line in which we can define the laser beams that we are going to use to represent each state:
```
idx_ranges = [89, 135, 179, 224, 269]
```
In that case the chosen laser beams corresponds to 89º, 135º, 179º, 224º, 269º. 

### Gazebo 9 Installation

To carry out this step, you should be placed in the following path: `python3_ws/src/` and execute this command:
```
   git clone -b kinetic-gazebo9 https://bitbucket.org/theconstructcore/turtlebot.git 
```
The resulting folder `turtlebot` has to be renamed to `kinetic-gazebo9` and the sub-folder `follow_line_tc_pkg` should be removed:
```
   rm -r kinetic_gazebo9/follow_line_tc_pkg
```
Next, the file `python3_ws/src/kinetic-gazebo9/turtlebot_gazebo/launch/includes/kobuki.launch` has to be modified so the tag `<arg name="urdf_file ...` looks like this:
```
   <arg name="urdf_file" default="$(find xacro)/xacro.py '$(find turtlebot_description)/robots/$(arg base)_$(arg stacks)_$(arg 3d_sensor).urdf.xacro'" />
```
To adapt the virtual lidar of Gazebo to the specifications of the physical lidar RPLidar A1 that we are going to use, you should modify the block tag `<gazebo reference="laser_sensor_link">` of the file `python3_ws/src/kinetic-gazebo9/kobuki_description/urdf/kobuki_gazebo.urdf.xacro` so it contains the following configuration data:
```
   <gazebo reference="laser_sensor_link">
		<sensor type="ray" name="laser_sensor">
			<pose>0 0 0 0 0 0</pose>
			<visualize>false</visualize>
			<update_rate>40</update_rate>
			<ray>
				<scan>
					<horizontal>
						<samples>360</samples>
						<resolution>1</resolution>
						<min_angle>-3.1241390705108643</min_angle>
						<max_angle>3.1415927410125732</max_angle>
					</horizontal>
				</scan>
				<range>
					<min>0.10</min>
					<max>12.0</max>
					<resolution>0.01</resolution>
				</range>
				<noise>
					<type>gaussian</type>
					<mean>0.0</mean>
					<stddev>0.01</stddev>
				</noise>
			</ray>
			<plugin name="gazebo_ros_head_hokuyo_controller" filename="libgazebo_ros_laser.so">
				<topicName>/kobuki/laser/scan</topicName>
				<frameName>laser_sensor_link</frameName>
			</plugin>
		</sensor>
	 </gazebo>
```

### Compile ROS and Gazebo packages

To accomplish this task, you should be placed in `python3_ws/` with the virtual environment `py3envros` activated and the file `/opt/ros/kinetic/setup.bash` loaded using the `source` command. Once you have done that, ROS and Gazebo packages can be compiled typing:
```
catkin_make -DPYTHON_EXECUTABLE:FILEPATH=$HOME/python3_ws/py3envros/bin/python
```
This action will create the folders `build` and `devel` inside `python3_ws`.
>**Important trick**:
>ROS works with Python 2.7 by default but as we want to use Pytorch with Python 3, we will need to compile ROS packages with the flag `-DPYTHON_EXECUTABLE:FILEPATH` to specify the path of the Python version that should be chosen.
After that, permissions will be granted to `python3_ws/src` folder:
```
   chmod 777 -R python3_ws/src/
```

## LOAD PYTHON AND ROS ENVIRONMENTS

Each time a new Linux Terminal is opened, you should activate the Python virtual environment and load the ROS configuration scripts with these commands:
```
   source $HOME/py3envros/bin/activate
   source /opt/ros/kinetic/setup.bash
   source $HOME/python3_ws/devel/setup.bash
```

## TRAIN DQN TO SOLVE MAZE ENVIRONMENT

Once Python and ROS environments have been loaded, you can train a model using Deep Q Learning algorithm so a virtual Turtlebot learns to navigate through a Maze environment rendered by OpenAI Gym and Gazebo, launching the following command:
```
   roslaunch turtle2_openai_ros_example start_training_maze_v2_dqn.launch
```
The checkpoints with the weights of the Policy Network will be saved in the folder `$HOME/python3_ws/src/turtle2_openai_ros_example/src/checkpoints/` and the events file for Tensorboard will be placed in the folder `$HOME/python3_ws/src/turtle2_openai_ros_example/src/logs/`.
The way to launch Tensorboard will be:
```
   tensorboard --logdir=$HOME/python3_ws/src/turtle2_openai_ros_example/src/logs/
```
The ROS launch file `$HOME/python3_ws/src/turtle2_openai_ros_example/launch/start_training_maze_v2_dqn.launch` will call the script `$HOME/python3_ws/src/turtle2_openai_ros_example/src/deepq.py`.

The script `deepq.py` is going to train a Deep Q Learning algorithm and to achive that it uses a Policy Network _Q_ and a Target Network _Q'_. 

Both Neural Networks will share the same topology, that will be a Multi Layer Perceptron. The input will be a state captured by the lidar that will be discretized (5 values for each reading by default) and the output will correspond to the possible actions (3 by default).

The Policy Network _Q_ will be updated each step with the following rule:
```
$$Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha * (R_{t+1} + \max_{a' \in A} Q'(s_{t+1}, a') - Q(s_{t}, a_{t})$$
```

<a href="https://www.codecogs.com/eqnedit.php?latex=Q(s_{t},&space;a_{t})&space;\leftarrow&space;Q(s_{t},&space;a_{t})&space;&plus;&space;\alpha&space;*&space;(R_{t&plus;1}&space;&plus;&space;\max_{a'&space;\in&space;A}&space;Q'(s_{t&plus;1},&space;a')&space;-&space;Q(s_{t},&space;a_{t})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(s_{t},&space;a_{t})&space;\leftarrow&space;Q(s_{t},&space;a_{t})&space;&plus;&space;\alpha&space;*&space;(R_{t&plus;1}&space;&plus;&space;\max_{a'&space;\in&space;A}&space;Q'(s_{t&plus;1},&space;a')&space;-&space;Q(s_{t},&space;a_{t})" title="Q(s_{t}, a_{t}) \leftarrow Q(s_{t}, a_{t}) + \alpha * (R_{t+1} + \max_{a' \in A} Q'(s_{t+1}, a') - Q(s_{t}, a_{t})" /></a>


The Target Network _Q'_ will copy the parameters of the Policy Network _Q_ periodically (we do that in the code each `target_update` steps which is configurable). It is set to 1000 steps by default.

The Loss function to optimize using gradient descent is:
```
$$L(w_{i}) = \mathbb{E}_{s_{t}, a_{t}, r_{t}, s_{t+1}}[(r_{t} + \gamma * \max_{a' \in A} Q'(s_{t+1}, a') - Q(s_{t}, a_{t}))^2]$$
```

<a href="https://www.codecogs.com/eqnedit.php?latex=L(w_{i})&space;=&space;\mathbb{E}_{s_{t},&space;a_{t},&space;r_{t&plus;1},&space;s_{t&plus;1}}[(r_{t&plus;1}&space;&plus;&space;\gamma&space;*&space;\max_{a'&space;\in&space;A}&space;Q'(s_{t&plus;1},&space;a')&space;-&space;Q(s_{t},&space;a_{t}))^2]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(w_{i})&space;=&space;\mathbb{E}_{s_{t},&space;a_{t},&space;r_{t&plus;1},&space;s_{t&plus;1}}[(r_{t&plus;1}&space;&plus;&space;\gamma&space;*&space;\max_{a'&space;\in&space;A}&space;Q'(s_{t&plus;1},&space;a')&space;-&space;Q(s_{t},&space;a_{t}))^2]" title="L(w_{i}) = \mathbb{E}_{s_{t}, a_{t}, r_{t+1}, s_{t+1}}[(r_{t+1} + \gamma * \max_{a' \in A} Q'(s_{t+1}, a') - Q(s_{t}, a_{t}))^2]" /></a>

Besides that, the `OneCycleLR` scheduler has been used with `max_lr = 0.01` and `total_steps = num_steps`.

This script is going to use the following settings:
```
# Hyperparameters
    gamma = 0.79  # initially 0.99 discount factor
    seed = 543  # random seed
    log_interval = 25  # controls how often we log progress, in episodes
    num_steps = 11e4  # number of steps to train on
    batch_size = 512  # batch size for optimization
    lr = 1e-3  # Default static learning rate but LR schedulers will be used
    eps_start = 1.0  # initial value for epsilon (in epsilon-greedy)
    eps_end = 0.1  # final value for epsilon (in epsilon-greedy)
    eps_decay = 8e4  # num_steps, length of epsilon decay, in env steps
    target_update = 1000  # how often to update target net, in env steps
    test_global_step = 0 # Global number of testing steps for tracking cummulative rewards in Tensorboard
```
The equation to compute the epsilon-greedy tradeoff will be:
```
eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)
```

## TRACKING RESULTS 

Tensorboard has been used as a tool to visualize relevant information related to the training and validation processes. To check the values of the Policy Neural Network's parameters and the gradients computed during Backpropagation, histograms will be shown.

Periodically (`i_episode % log_interval == 0 or step_count >= num_steps`), the Policy Network will be evaluated calling the `test` method in `deepq.py`. In that function, an episode will be run using the Policy Network in eval mode and the cumulated rewards for this episode will be shown in Tensorboard and it will be possible to compare them with the cumulated rewards of previous epochs.

The events files of Tensorboard will be stored in `$HOME/python3_ws/src/turtle2_openai_ros_example/src/logs/<YYYYMMDD-HHmmss>/`.

Besides that, as a way to gather more details, a Python `namedtuple` called `Trace` (defined in the script `memory.py`) will collect the following data: 
```
self.Trace = namedtuple('Trace', ('episode', 'step', 'st', 'act', 'next_st', 'rw', 'policy_act', 'epsi', 'policy_used'))
```

The description of these fields is:
* `episode`: The number of the episode.
* `step`: The number of the step of that episode.
* `st`: The values of the state with the following format \[d_obs_89º, d_obs_135º, d_obs_179º, d_obs_224º, d_obs_269º] where d_obs_89º stands for the distance with an obstacle in the direction of the 89º laser beam.
* `act`: It is the action taken in that of step of that episode. As we are going to play with the epsilon-greedy trade-off, it's interesting to know what action was taken.
* `next_st`: It's the next state that the agent will find when it takes the action `act` from the state `st` in the step `step` of the episode `episode`. It follows the same format as `st`.
* `rw`: It is the reward achieved taken the action `act` from the state `st`.
* `policy_act`: It stores the action predicted by the Policy Network when it receives the state `st`. This field allows us to know what action the Policy Network would predict although the final action taken was a random action because of the epsilon-greedy trade-off.
* `epsi`: It contains tha value of epsilon in the step `step` of the episode `episode`.
* `policy_used`: It's a boolean value so if it's True the tha action taken in the step `step` of the episode `episode` was chosen by the Policy Network.

An example of this data structure is:
```
[{"episode": 0, "step": 1, "st": [0.8, 1.1, 2.5, 1.1, 0.8], "act": 0, "next_st": [0.7, 1.1, 2.5, 1.1, 0.7], "rw": 5, "policy_act": 2, "epsi": 1.0, "policy_used": false}, {"episode": 0, "step": 2, "st": [0.7, 1.1, 2.5, 1.1, 0.7], "act": 1, "next_st": [0.7, 1.2, 2.4, 1.0, 0.7], "rw": 4, "policy_act": 2, "epsi": 0.9999887500703122, "policy_used": false}, ...]
```

The content of this data structure will be exported to contiguous json files located at:
```
$HOME/python3_ws/src/turtle2_openai_ros_example/src/trace/
```

So each file will have `target_update` (1000 by default) tuples corresponding to this amount of steps and the next json file will store the following `target_update` tuples. We have done that so each file can be opened and visualized without memory issues.  



## TEST THE TRAINED MODEL IN THE OPENAI GYM AND GAZEBO ENVIRONMENT

Once the model has been trained, you can test it in the Maze environment offered by OpenAI Gym and Gazebo executing the following ROS node:
```
roslaunch turtle2_openai_ros_example start_test_maze_v2_dqn.launch
```
The Maze enviroment with the Turtlebot virtual agent will be shown through the Gazebo simulator. It's an intuitive and visual way to show the behavior of the policy.

The launch file `$HOME/python3_ws/src/turtle2_openai_ros_example/launch/start_test_maze_v2_dqn.launch` calls the script `$HOME/python3_ws/src/turtle2_openai_ros_example/src/test_deepq.py` which loads the parameters of the trained policy network and runs a number `n_epochs` of testing epochs. The relevant configurable parameters are:
```
MODEL_PATH = '$HOME/python3_ws/src/turtle2_openai_ros_example/src/checkpoints/dqn-final-episode-2671-step-110007.pt'
n_epochs = 20
logdir = os.path.join("$HOME/python3_ws/src/turtle2_openai_ros_example/src/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=logdir)
```
The cumulative reward of the `n_epochs` testing epochs is tracked in the `logdir` events file and it can be inspected using Tensorboard.


## DEPLOY THE TRAINED MODEL IN A REAL WORLD SCENARIO USING PHYSICAL TURTLEBOT

In this step, we will have to consider 2 machines: 
1. The Turtlebot side that will have an Intel NUC i7 processor, a Kobuki mobile platform and a RPLidar A1 version.
2. The laptop side where the trained model and the deploy script are stored.
Both of them will have to be reachable through the network.

### Turtlebot side

#### Prerequisites

The Intel NUC i7 processor will have Ubuntu 16.04 as OS and you will have to install ROS Kinetic. Once you have done that, it will be necessary to install and compile the ROS package called `rplidar_ros`. To achive that, you have to execute the following commands in a Terminal of the Intel NUC:
```
   mkdir -p $HOME/catkin_ws/src
   cd $HOME/catkin_ws/src/
   git clone https://github.com/robopeak/rplidar_ros.git
   cd ..
   catkin_make
   source devel/setup.bash 
   ls -l /dev |grep ttyUSB
   sudo chmod 666 /dev/ttyUSB0
   cd src/rplidar_ros/scripts/
   ./create_udev_rules.sh 
   ls -l /dev |grep ttyUSB
   sudo adduser $USER dialout
```
Once you have change the USB port remap, you can change the launch file of `rplidar_ros` about the serial_port value so it looks like this:
```
<param name="serial_port" type="string" value="/dev/rplidar"/>
```

#### Launching `roscore` and `rplidar_ros` packages

At this point, all the required software is installed in the Intel NUC (the processor of the robot). You will have to open 2 Linux Terminals to launch the ROS Master node `roscore` and the lidar package called `rplidar_ros`. As this machine will execute the ROS Master node, it will be necessary to specify the IP of this workstation in the  the ROS environment variable `ROS_IP`.
1. Terminal 1:
```
source catkin_ws/devel/setup.bash
export ROS_IP=192.168.x.x
roscore
```
2. Terminal 2:
```
source catkin_ws/devel/setup.bash
export ROS_IP=<IP_Turtlebot>
roslaunch rplidar_ros rplidar.launch
```

### Laptop side

In this side you are going to launch the deploy script which will load the trained model to predict the next action using the state captured by the physical RPLidar A1. 
The way to allow that this machine is connected with the ROS Master node in the Turtlebot and can subscribe to the topic where the RPLidar is publishing its captured data will be setting the `ROS_MASTER_URI` variable environment with the IP of the Intel NUC and the `ROS_IP` with the IP of this laptop.
```
   source $HOME/python3_ws/py3envros/bin/activate
   source /opt/ros/kinetic/setup.bash
   source $HOME/python3_ws/devel/setup.bash
   export ROS_MASTER_URI=http://<IP_Turtlebot>:11311
   export ROS_IP=<IP_Laptop>
   roslaunch turtle2_openai_ros_example rl_robot.launch
```
The launch file `$HOME/python3_ws/src/turtle2_openai_ros_example/launch/rl_robot.launch` will call the script `$HOME/python3_ws/src/turtle2_openai_ros_example/src/deploy_robot.py` using the parameters specified in the file `$HOME/python3_ws/src/turtle2_openai_ros_example/launch/rl_robot_params.yaml`.

Then you should see how the Turtlebot navigates using the states gathered by the RPLidar and predicting the next action using the trained Policy Neural Network. Find a safe place!! :P

   


   
   


        
    
   


   


        
        
   




