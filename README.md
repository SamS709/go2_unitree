# Depolying a policy on the Unitree Go2 using unitree_python_sdk2

## Installation


```bash
mkdir unitree
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
# HERE follow the instructions provided by unitree_sdk2_python repo to make the python install ('pip install -e .' has to work)
cd ..
git clone https://github.com/SamS709/go2_unitree.git
```

## Run

### Controller

The trained policy requires an input velocity and height, which are sent by a controller.

Test the controller and note which buttons are RT, LT, A, LB which are used in the controller.


```bash
cd go2_unitree
python controller/controller.py
# in an other terminal (same location: go2_unitree/controller)
python test_controller.py
```
The output should be the following (1 when a button is pressed and rx, rl, lx, ... floats varying when you move the joysticks.)

Check that this basic exemple provided by unitree works on your robot:

```bash
cd unitree_sdk2_python/example/go2/low_level
python go2_stand_example.py
```

If it doesn't it might be because of the motion switcher hich is not avaible on old versions of the firmware. There is still a hope that it works later with the policy because I don't use that class.

### Policy

CONNECT YOUR ROBOT

Before launching the policy, here are the controls:

- Left joystick controls x and y velocities
- Right joystick controls yaw velocity (along z axis) and the height of the robot
- RT button: activates emergency mode: the robot stops, setting kp and kd progressively to zero: mashmallow mode
- LT button: activates the policy
- A button: the robot sits down

Note that after the robot sat, it is not possible to lauch the policy anymore. In this case, just kill the terminal and rerun the programm.

In one terminal run:

```bash
python controller/controller.py
```

In an other terminal run:

```bash
python go2/go2_publisher.py
```