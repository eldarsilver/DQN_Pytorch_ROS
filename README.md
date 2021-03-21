# DQN_Pytorch_ROS

The goal of this project is to train Reinforcement Learning algorithms using Pytorch in simulated environments rendered by OpenAI Gym and Gazebo, controlling the agent through ROS (Robot Operating System). Finally, the trained models will be deployed in a real world scenario using the robot called Turtlebot.  

## INSTALLATION

This project needs to be installed and executed in Ubuntu 16.04 because ROS Kinetic will be used to manage the robot.

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
   sudo apt install python-rosdep
   sudo rosdep init
   source /opt/ros/kinetic/setup.bash
   sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
   sudo apt install python-rosdep
   sudo apt-get install rospkg    
```

### Clone this repository 

The repository should be cloned under the folder `python3_ws/src/`:
```
   git clone <repository_url>
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
        requirements.txt
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

### Load Python and ROS environments

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

## DEPLOY THE TRAINED MODEL IN A REAL WORLD SCENARIO USING PHYSICAL TURTLEBOT

In this step, we will have to consider 2 machines: 
1. The Turtlebot side that will have a Intel NUC i7 processor, a Kobuki mobile platform and a RPLidar A1 version.
2. The laptop side where the trained model and the deploy script are stored.
Both of them will have to be reachable through the network.

### Turtlebot side

#### Prerequisites

The Intel NUC i7 processor will have Ubuntu 16.04 as OS and you sill have to install ROS Kinetic. Once you have done that, it will be necessary to install and compile the ROS package called `rplidar_ros`. To achive that, you have to execute the following commands in a Terminal of the Intel NUC:
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
The way to allow that this machine is connected with the ROS Maste node in the Turtlebot and can subscribe to the topic where the RPLidar is publishing its captured data will be setting the `ROS_MASTER_URI` variable environment with the IP of the Intel NUC and the `ROS_IP` with the IP of this laptop.
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

   


   
   


        
    
   


   


        
        
   




