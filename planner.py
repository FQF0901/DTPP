import math
import time
import matplotlib.pyplot as plt
from shapely import Point, LineString
from planner_utils import *
from obs_adapter import *
from trajectory_tree_planner import TreePlanner
from scenario_tree_prediction import *

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring

'''
1. DetectionsTracks
    DetectionsTracks 是一种观测类型（observation type），表示从传感器数据中检测到的动态物体（如车辆、行人等）及其轨迹信息。它通常包含以下内容：

    detections：当前帧中检测到的物体。
    tracks：这些物体的历史轨迹信息。
    在路径规划过程中，DetectionsTracks 提供了环境中的动态障碍物信息，帮助规划器避免碰撞并做出合理的决策。

2. AbstractPlanner
    AbstractPlanner 是一个抽象基类，定义了所有路径规划器必须实现的接口。它确保所有具体的规划器都遵循相同的结构和行为规范。具体来说，AbstractPlanner 包含以下几个关键方法：

    name()：返回规划器的名称。
    observation_type()：指定规划器期望的观测类型（例如 DetectionsTracks）。
    initialize()：初始化规划器，通常在这个阶段加载地图信息、目标位置等。
    compute_trajectory()：根据当前的观测和历史信息计算未来的轨迹。
    通过继承 AbstractPlanner，具体的规划器可以实现自己的逻辑，同时保持与整个系统的兼容性。

3. PlannerInitialization
    PlannerInitialization 是一个数据类，包含了规划器初始化时需要的信息。它通常包括以下内容：

    map_api：地图API，提供了访问地图数据的接口，如道路、车道等信息。
    mission_goal：任务的目标位置或终点。
    route_roadblock_ids：规划路径上的一系列路障ID，用于构建路径规划的基础。
    这些信息在规划器初始化时提供，确保规划器能够正确地理解环境和任务需求。

4. PlannerInput
    PlannerInput 是一个数据类，包含了每次调用 compute_trajectory() 方法时需要的输入信息。它通常包括以下内容：

    iteration：当前迭代次数，用于跟踪时间步。
    history：历史状态记录，包括过去的自车状态和观测信息。
    traffic_light_data：交通信号灯的状态信息，帮助规划器了解当前的交通规则。
    这些信息在每次规划时提供，确保规划器能够基于最新的环境信息进行决策。

5. InterpolatedTrajectory
    InterpolatedTrajectory 是一个轨迹对象，表示由一系列离散状态点插值得到的连续轨迹。它通常用于将规划器生成的离散轨迹点转换为平滑的、可执行的路径。具体来说，它包含以下功能：

    states：一系列离散的状态点，每个状态点包含位置、速度、加速度等信息。
    interpolation：通过插值算法（如线性插值、样条插值等）将离散状态点连接成一条平滑的轨迹。
    在路径规划过程中，InterpolatedTrajectory 将规划器生成的离散轨迹点转换为可供执行的实际路径，确保车辆能够平稳地沿着规划路径行驶。
'''
class Planner(AbstractPlanner):
    def __init__(self, model_path, device):
        self._future_horizon = T # [s] 
        self._step_interval = DT # [s]
        self._N_points = int(T/DT)
        self._model_path = model_path
        self._device = device

    def name(self) -> str:
        return "DTPP Planner"
    
    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal    # 务的目标位置或终点
        self._route_roadblock_ids = initialization.route_roadblock_ids  # 规划路径上的一系列路障ID
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        self._trajectory_planner = TreePlanner(self._device, self._encoder, self._decoder)

    def _initialize_model(self):
        model = torch.load(self._model_path, map_location=self._device)
        self._encoder = Encoder()
        self._encoder.load_state_dict(model['encoder'])
        self._encoder.to(self._device)
        self._encoder.eval()
        self._decoder = Decoder()
        self._decoder.load_state_dict(model['decoder'])
        self._decoder.to(self._device)
        self._decoder.eval()

    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]

    def compute_planner_trajectory(self, current_input: PlannerInput):
        # Extract iteration, history, and traffic light
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state

        # Construct input features
        start_time = time.perf_counter()
        features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        # Get starting block
        starting_block = None
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break
        
        # Get traffic light lanes
        traffic_light_lanes = []
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                traffic_light_lanes.append(lane_conn)

        # Tree policy planner
        try:
            plan = self._trajectory_planner.plan(iteration, ego_state, features, starting_block, self._route_roadblocks, 
                                             self._candidate_lane_edge_ids, traffic_light_lanes, observation)
        except Exception as e:
            print("Error in planning")
            print(e)
            plan = np.zeros((self._N_points, 3))
            
        # Convert relative poses to absolute states and wrap in a trajectory object
        states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)
        print(f'Step {iteration+1} Planning time: {time.perf_counter() - start_time:.3f} s')

        return trajectory