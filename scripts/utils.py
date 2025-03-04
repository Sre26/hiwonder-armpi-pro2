from typing import List
from dataclasses import dataclass, field
import math, numpy as np
from math import sqrt, sin, cos, atan, atan2
from functools import singledispatch

PI = 3.1415926535897932384

@dataclass
class State:
    """This dataclass represents the system state (pos and vel) """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    x_dot: float = 0.0
    y_dot: float = 0.0
    theta_dot: float = 0.0


@dataclass
class Controls:
    """This dataclass represents the system controls """
    v: float = 0.0
    w: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class GamepadCmds:
    """This dataclass represents the gamepad commands """
    base_vx: int = 0
    base_vy: int = 0
    base_w: int = 0
    arm_vx: int = 0
    arm_vy: int = 0
    arm_vz: int = 0
    arm_j1: int = 0
    arm_j2: int = 0
    arm_j3: int = 0
    arm_j4: int = 0
    arm_j5: int = 0
    arm_ee: int = 0
    arm_home: int = 0

def print_dataclass(obj):
    print("------------------------------------")
    for field in obj.__dataclass_fields__:
        print(f"{field}: {round(getattr(obj, field), 3)}")
    print("------------------------------------ \n")


class EndEffector:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rotx: float = 0.0
    roty: float = 0.0
    rotz: float = 0.0


def rotm_to_euler(R) -> tuple:
    """Converts a rotation matrix to Euler angles (roll, pitch, yaw).

    Args:
        R (np.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: Roll, pitch, and yaw angles (in radians).
    
    Reference:
        Based on the method described at:
        https://motion.cs.illinois.edu/software/klampt/latest/pyklampt_docs/_modules/klampt/math/so3.html
    """
    r31 = R[2,0] # -sin(p)
    r11 = R[0,0] # cos(r)*cos(p)
    r33 = R[2,2] # cos(p)*cos(y)
    r12 = R[0,1] # -sin(r)*cos(y) + cos(r)*sin(p)*sin(y)

    # compute pitch
        # condition r31 to the range of asin [-1, 1]
    r31 = min(1.0, max(r31, -1.0))
    p = -math.asin(r31)
    cosp = math.cos(p)

    if abs(cosp) > 1e-7:
        cosr = r11 / cosp
        # condition cosr to the range of acos [-1, 1]
        cosr = min(1.0, max(cosr, -1.0))
        r = math.acos(cosr)

        cosy = r33 / cosp
        # condition cosy to the range of acos [-1, 1]
        cosy = min(1.0, max(cosy, -1.0))
        y = math.acos(cosy)
    
    else:
        # pitch (p) is close to 90 deg, i.e. cos(p) = 0.0
        # there are an infinitely many solutions, so we set y = 0
        y = 0
        # r12: -sin(r)*cos(y) + cos(r)*sin(p)*sin(y) -> -sin(r)
            # condition r12 to the range of asin [-1, 1]
        r12 = min(1.0, max(r12, -1.0))
        r = -math.asin(r12)
    
    
    r11 = R[0,0] if abs(R[0,0]) > 1e-7 else 0.0
    r21 = R[1,0] if abs(R[1,0]) > 1e-7 else 0.0
    r32 = R[2,1] if abs(R[2,1]) > 1e-7 else 0.0
    r33 = R[2,2] if abs(R[2,2]) > 1e-7 else 0.0
    r31 = R[2,0] if abs(R[2,0]) > 1e-7 else 0.0

    # print(f"R : {R}")

    if r32 == r33 == 0.0:
        # print("special case")
        # pitch is close to 90 deg, i.e. cos(pitch) = 0.0
        # there are an infinitely many solutions, so we set yaw = 0
        pitch, yaw = PI/2, 0.0
        # r12: -sin(r)*cos(y) + cos(r)*sin(p)*sin(y) -> -sin(r)
            # condition r12 to the range of asin [-1, 1]
        r12 = min(1.0, max(r12, -1.0))
        roll = -math.asin(r12)
    else:
        yaw = math.atan2(r32, r33)        
        roll = math.atan2(r21, r11)
        denom = math.sqrt(r11 ** 2 + r21 ** 2)
        pitch = math.atan2(-r31, denom)

    return roll, pitch, yaw


def dh_to_matrix(dh_params: list) -> np.ndarray:
    """Converts Denavit-Hartenberg parameters to a transformation matrix.

    Args:
        dh_params (list): Denavit-Hartenberg parameters [theta, d, a, alpha].

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    theta, d, a, alpha = dh_params
    return np.array([
        [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
        [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])


def euler_to_rotm(rpy: tuple) -> np.ndarray:
    """Converts Euler angles (roll, pitch, yaw) to a rotation matrix.

    Args:
        rpy (tuple): A tuple of Euler angles (roll, pitch, yaw).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rpy[2]), -math.sin(rpy[2])],
                    [0, math.sin(rpy[2]), math.cos(rpy[2])]])
    R_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                    [0, 1, 0],
                    [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
    R_z = np.array([[math.cos(rpy[0]), -math.sin(rpy[0]), 0],
                    [math.sin(rpy[0]), math.cos(rpy[0]), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x


@dataclass
class SimData:
    """Captures simulation data for storage.

    Attributes:
        x (List[float]): x-coordinates over time.
        y (List[float]): y-coordinates over time.
        theta (List[float]): Angles over time.
        x_dot (List[float]): x-velocity over time.
        y_dot (List[float]): y-velocity over time.
        theta_dot (List[float]): Angular velocity over time.
        v (List[float]): Linear velocity over time.
        w (List[float]): Angular velocity over time.
        vx (List[float]): x-component of linear velocity over time.
        vy (List[float]): y-component of linear velocity over time.
    """
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    theta: List[float] = field(default_factory=list)
    x_dot: List[float] = field(default_factory=list)
    y_dot: List[float] = field(default_factory=list)
    theta_dot: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    w: List[float] = field(default_factory=list)
    vx: List[float] = field(default_factory=list)
    vy: List[float] = field(default_factory=list)


def check_joint_limits(theta: List[float], theta_limits: List[List[float]]) -> bool:
    """Checks if the joint angles are within the specified limits.

    Args:
        theta (List[float]): Current joint angles.
        theta_limits (List[List[float]]): Joint limits for each joint.

    Returns:
        bool: True if all joint angles are within limits, False otherwise.
    """
    for i, th in enumerate(theta):
        if not (theta_limits[i][0] <= th <= theta_limits[i][1]):
            return False
    return True


def calc_distance(p1: State, p2: State) -> float:
    """Calculates the Euclidean distance between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The Euclidean distance between p1 and p2.
    """
    return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def calc_heading(p1: State, p2: State) -> float:
    """Calculates the heading (angle) between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The heading angle in radians.
    """
    return atan2(p1.y - p2.y, p1.x - p2.x)


@singledispatch
def calc_angdiff(p1: State, p2: State) -> float:
    """Calculates the angular difference between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The angular difference in radians.
    """
    d = p1.theta - p2.theta
    return math.fmod(d, 2 * math.pi)


@calc_angdiff.register
def _(th1: float, th2: float) -> float:
    """Calculates the angular difference between two angles.

    Args:
        th1 (float): The first angle.
        th2 (float): The second angle.

    Returns:
        float: The angular difference in radians.
    """
    return math.fmod(th1 - th2, 2 * math.pi)


def near_zero(arr: np.ndarray) -> np.ndarray:
    """Checks if elements of an array are near zero.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        np.ndarray: An array with zeros where values are near zero, otherwise the original values.
    """
    tol = 1e-6
    return np.where(np.isclose(arr, 0, atol=tol), 0, arr)

class FiveDOFRobot:
    """
    A class to represent a 5-DOF robotic arm with kinematics calculations, including
    forward kinematics, inverse kinematics, velocity kinematics, and Jacobian computation.

    Attributes:
        l1, l2, l3, l4, l5: Link lengths of the robotic arm.
        theta: List of joint angles in radians.
        theta_limits: Joint limits for each joint.
        ee: End-effector object for storing the position and orientation of the end-effector.
        num_dof: Number of degrees of freedom (5 in this case).
        points: List storing the positions of the robot joints.
        DH: Denavit-Hartenberg parameters for each joint.
        T: Transformation matrices for each joint.
    """
    
    def __init__(self):
        """Initialize the robot parameters and joint limits."""
        # Link lengths
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.155, 0.099, 0.095, 0.055, 0.105
        
        # Joint angles (initialized to zero)
        self.theta = [0, 0, 0, 0, 0]
        
        # Joint limits (in radians)
        self.theta_limits = [
            [-np.pi, np.pi], 
            [-np.pi/3, np.pi], 
            [-np.pi+np.pi/12, np.pi-np.pi/4], 
            [-np.pi+np.pi/12, np.pi-np.pi/12], 
            [-np.pi, np.pi]
        ]
        
        # End-effector object
        self.ee = EndEffector()
        
        # Robot's points
        self.num_dof = 5
        self.points = [None] * (self.num_dof + 1)

        # Denavit-Hartenberg parameters and transformation matrices
        self.DH = [
            [self.theta[0],       self.l1,            0,          180/2],  # 0H1
            [self.theta[1]+180/2,  0,                  self.l2,    180],    # 1H2
            [self.theta[2],       0,                  self.l3,    180],    # 2H3
            [self.theta[3]+180/2,  0,                  0,          180/2],  # 3H4
            [self.theta[4],       self.l4 + self.l5,  0,          0,],    # 4H5
        ]
        
        # container for successive transformation matrices (ie 2H3, 3H4, ...)
        self.T = np.stack(
            [
                dh_to_matrix([self.theta[0],       self.l1,            0,          180/2]),  # 0H1
                dh_to_matrix([self.theta[1]+180/2,  0,                  self.l2,    180]),    # 1H2
                dh_to_matrix([self.theta[2],       0,                  self.l3,    180]),    # 2H3
                dh_to_matrix([self.theta[3]+180/2,  0,                  0,          180/2]),  # 3H4
                dh_to_matrix([self.theta[4],       self.l4 + self.l5,  0,          0,]),    # 4H5

            ], axis=0)

    
    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate forward kinematics based on the provided joint angles.
        
        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """
        # update transformaiton matrices to use the thetas in args

        # check that theta values are in radians
        theta = np.array(theta)
        if radians == False:
            # if values arent in radians, then convert
            theta = theta * PI/180

        # update transformation matrices with the new theta vals
        self.T[0, :, :] = dh_to_matrix([theta[0],       self.l1,            0,          180/2])  # 0H1
        self.T[1, :, :] = dh_to_matrix([theta[1]+180/2,  0,                  self.l2,    180])    # 1H2
        self.T[2, :, :] = dh_to_matrix([theta[2],       0,                  self.l3,    180])    # 2H3
        self.T[3, :, :] = dh_to_matrix([theta[3]+180/2,  0,                  0,          180/2])  # 3H4
        self.T[4, :, :] = dh_to_matrix([theta[4],       self.l4 + self.l5,  0,          0,])    # 4H5

        # Calculate robot points (positions of joints)
        self.calc_robot_points()


    
    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate the joint velocities required to achieve the given end-effector velocity.
        
        Args:
            vel: Desired end-effector velocity (3x1 vector).
        """
        # calculate the jacobian for the desired EE velocity
        Jacobian_v = self.make_Jacobian_v(vel)
        
        time_step = 0.01    # choose arbitrary timestep fro RRMC

        # calc angular velocities for joints
        inv_Jv = np.linalg.pinv(Jacobian_v)
        theta_dot = inv_Jv @ vel

        # update self.theta
        self.theta = self.theta + (time_step * theta_dot)

        # Recompute robot points based on updated joint angles
        self.calc_forward_kinematics(self.theta, radians=False)

    
    def make_Jacobian_v(self, vel: list):
        """ 
        Computes the linear component of the Jacobian, Jacobian_v, via
        the geometric approach. 
        
        This is/can be used for J_inv @ vel = theta_dot where theta_dot are 
        the joint velocities corresponding to vel, the desired EE velocity.

        Args:
            vel: Desired end-effector velocity (3x1 vector).
        Returns:
            Jacobian_v: the linear component of the Jacobian (3x5 matrix)
        """
        # -----------------------------------------------------
        # PROCESS: the geometric process for calculating the  
        # linear component of the jacobian is as follows:

        # vel_EE = Jacobian_v @ theta_dot
        # vel_EE = x_dot, y_dot, z_dot (dot indicates time deriv.)
        # theta_dot is a vector of the time deriv. of each theta

        # Jacobian_v = z_vec X r_vec (cross product)
        
        # z_vec is the z axis of the current joint in reference
        # to Frame 0
        # (z_vec calculated as z_0 @ R_0i, where z_0 is the z_axis at 
        # frame 0 and R_0i is the rotation matrix from frame 0 to 
        # frame i, extracted from the HTM 0Hi)

        # r_vec is the distance from the current joint to the EE
        # (r_vec calculated as r_EE - r_i, where r_EE is the dist. from
        # joint 1 to the EE, and r_i is the dist from joint 1 to joint i)
        # -----------------------------------------------------

       # compute the cumulative transformation matrix
        cum_htm = [np.eye(4)]
        for i in range(self.num_dof):
            cum_htm.append(cum_htm[-1] @ self.T[i, :, :])

        # radius from base (Frame 0) to EE (Frame 5 in this case)
        r_EE = cum_htm[-1][0:3, 3]

        # get z vectors from the cumulative transformation matrices
        # the z vectors are the third colum of the rotation matrices
        z_vec = np.zeros((3, self.num_dof))
        for j in range(self.num_dof):
            z_vec[:, j] = cum_htm[j][0:3, 2].T

        # make an array of r_vectors
        # zero array to store distance (3x1 vector) for each cumulative radius (ie 0-5, 1-5, ...) 
        r_vec = np.zeros((3, self.num_dof))

        r_vec[:, 0] = r_EE
        for k in range(self.num_dof):
            # compute new r vector (ex r1-5, r2-5, ...) as r_EE - r0-i 
            # extract r0-i from the fourth column of the cumulative htm matrices
            r_vec[:, k] = (r_EE - cum_htm[k][0:3, 3])

        # calculate the Jacobian terms
        # container for the linear velocity component of the Jacobian 
        J_v = np.zeros((3, self.num_dof))

        for i1 in range(self.num_dof):
            # column of J = z x r (cross product)
            J_v[:, i1] = np.cross(z_vec[:, i1], r_vec[:, i1])
        
        return J_v
