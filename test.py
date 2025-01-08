import yaml
import datetime
import torch
import argparse
import warnings
from tqdm import tqdm
from planner import Planner
from common_utils import *
warnings.filterwarnings("ignore") 

from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
from nuplan.planning.simulation.main_callback.metric_summary_callback import MetricSummaryCallback
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.nuboard.base.data_class import NuBoardFile


def build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir):
    """
    Builds the main experiment folder for simulation.
    :return: The main experiment folder path.
    """
    print('Building experiment folders...')

    exp_folder = pathlib.Path(output_dir)
    print(f'\nFolder where all results are stored: {exp_folder}\n')
    exp_folder.mkdir(parents=True, exist_ok=True)

    # Build nuboard event file.
    nuboard_filename = exp_folder / (f'nuboard_{int(time.time())}' + NuBoardFile.extension())
    nuboard_file = NuBoardFile(
        simulation_main_path=str(exp_folder),
        simulation_folder=simulation_dir,
        metric_main_path=str(exp_folder),
        metric_folder=metric_dir,
        aggregator_metric_folder=aggregator_metric_dir,
    )

    metric_main_path = exp_folder / metric_dir
    metric_main_path.mkdir(parents=True, exist_ok=True)

    nuboard_file.save_nuboard_file(nuboard_filename)
    print('Building experiment folders...DONE!')

    return exp_folder.name


def build_simulation(experiment, planner, scenarios, output_dir, simulation_dir, metric_dir):
    runner_reports = []
    print(f'Building simulations from {len(scenarios)} scenarios...')

    metric_engine = build_metrics_engine(experiment, output_dir, metric_dir)
    print('Building metric engines...DONE\n')

    # Iterate through scenarios
    for scenario in tqdm(scenarios, desc='Running simulation'):
        # Ego Controller and Perception
        if experiment == 'open_loop_boxes':
            ego_controller = LogPlaybackController(scenario) 
            observations = TracksObservation(scenario)
        elif experiment == 'closed_loop_nonreactive_agents':
            tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], 
                                 r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10, 
                                 jerk_penalty=1e-4, curvature_rate_penalty=1e-2, 
                                 stopping_proportional_gain=0.5, stopping_velocity=0.2)
            motion_model = KinematicBicycleModel(get_pacifica_parameters())
            ego_controller = TwoStageController(scenario, tracker, motion_model) 
            observations = TracksObservation(scenario)
        elif experiment == 'closed_loop_reactive_agents':      
            tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], 
                                 r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10, 
                                 jerk_penalty=1e-4, curvature_rate_penalty=1e-2, 
                                 stopping_proportional_gain=0.5, stopping_velocity=0.2)
            motion_model = KinematicBicycleModel(get_pacifica_parameters())
            ego_controller = TwoStageController(scenario, tracker, motion_model) 
            observations = IDMAgents(target_velocity=10, min_gap_to_lead_agent=1.0, headway_time=1.5,
                                     accel_max=1.0, decel_max=2.0, scenario=scenario,
                                     open_loop_detections_types=["PEDESTRIAN", "BARRIER", "CZONE_SIGN", "TRAFFIC_CONE", "GENERIC_OBJECT"])
        else:
            raise ValueError(f"Invalid experiment type: {experiment}")
            
        # Simulation Manager
        simulation_time_controller = StepSimulationTimeController(scenario)

        # Stateful callbacks
        metric_callback = MetricCallback(metric_engine=metric_engine)
        sim_log_callback = SimulationLogCallback(output_dir, simulation_dir, "msgpack")

        # Construct simulation and manager
        simulation_setup = SimulationSetup(
            time_controller=simulation_time_controller,
            observations=observations,
            ego_controller=ego_controller,
            scenario=scenario,
        )

        simulation = Simulation(
            simulation_setup=simulation_setup,
            callback=MultiCallback([metric_callback, sim_log_callback])
        )

        # Begin simulation
        simulation_runner = SimulationRunner(simulation, planner)
        report = simulation_runner.run()
        runner_reports.append(report)
    
    # save reports
    save_runner_reports(runner_reports, output_dir, 'runner_reports')

    # Notify user about the result of simulations
    failed_simulations = str()
    number_of_successful = 0

    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            print("Failed Simulation.\n '%s'", result.error_message)
            failed_simulations += f"[{result.log_name}, {result.scenario_name}] \n"

    number_of_failures = len(scenarios) - number_of_successful
    print(f"Number of successful simulations: {number_of_successful}")
    print(f"Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    if number_of_failures > 0:
        print(f"Failed simulations [log, token]:\n{failed_simulations}")
    
    print('Finished running simulations!')

    return runner_reports


def build_nuboard(scenario_builder, simulation_path):
    nuboard = NuBoard(
        nuboard_paths=simulation_path,
        scenario_builder=scenario_builder,
        vehicle_parameters=get_pacifica_parameters(),
        port_number=5006
    )

    nuboard.run()


def main(args):
    """
    主函数，用于执行测试模拟并生成度量和汇总报告。
    
    参数:
    - args: 命令行参数，包含测试类型、模型路径、设备信息等。

    nuplan func:
    - MetricAggregatorCallback: 用于在仿真过程中收集和聚合度量指标。它可以在仿真运行时或结束时计算和存储各种性能指标。
    - MetricFileCallback: 用于将度量指标保存到文件中。它通常在仿真结束时被调用，将收集到的度量数据写入指定的文件格式（如JSON或CSV）。
    - MetricSummaryCallback: 用于生成度量指标的摘要报告。它可以在仿真结束时创建一个简要的报告，概述仿真运行的关键性能指标。
    - MultiMainCallback: 用于管理多个回调函数。它允许在仿真过程中同时运行多个回调函数，以便执行多种操作（如记录日志、保存度量数据等）。
    - on_run_simulation_start: 这是一个回调函数，在仿真开始时被调用。它可以用于初始化仿真环境、设置初始条件或执行其他启动时的操作。
    - ScenarioMapping: 用于将仿真场景映射到特定的配置或参数集。它可以根据输入数据或配置文件生成相应的仿真场景。
    - NuPlanScenarioBuilder: 用于构建仿真场景。它根据输入的配置和参数生成具体的仿真场景，包括场景中的车辆、道路、交通信号等元素
    - ScenarioFilter: 用于过滤仿真场景的类。它可以根据特定的条件（如场景类型、时间、天气等）筛选出符合条件的场景。
    - SingleMachineParallelExecutor: 用于在单台机器上并行执行仿真任务的类。它可以利用多线程或多进程来加速仿真任务的执行。
    - get_scenarios: 用于获取仿真场景的函数。它通常会从某个数据源（如数据库或文件系统）中加载场景数据，并返回一个场景列表。
    - on_run_simulation_end: 在仿真结束时调用的回调函数。它可以用于执行一些清理工作、记录结果或触发后续的处理步骤。
    """
    # parameters
    experiment_name = args.test_type  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    job_name = 'DTPP_planner'
    experiment_time = datetime.datetime.now()
    experiment = f"{experiment_name}/{job_name}/{experiment_time}"  
    output_dir = f"testing_log/{experiment}"
    simulation_dir = "simulation"
    metric_dir = "metrics"  # 设置度量目录名称为'metrics'
    aggregator_metric_dir = "aggregator_metric" # 设置聚合度量目录名称为'aggregator_metric'

    # initialize planner
    torch.set_grad_enabled(False)   # 关闭梯度计算
    planner = Planner(model_path=args.model_path, device=args.device)

    # initialize main aggregator
    metric_aggregators = build_metrics_aggregators(experiment_name, output_dir, aggregator_metric_dir)  # 构建指标聚合器，用于汇总和处理实验中产生的各种指标
    metric_save_path = f"{output_dir}/{metric_dir}"
    metric_aggregator_callback = MetricAggregatorCallback(metric_save_path, metric_aggregators) # 创建指标聚合回调，用于在特定事件中处理指标数据聚合
    metric_file_callback = MetricFileCallback(metric_file_output_path=f"{output_dir}/{metric_dir}", # 创建指标文件回调，用于在特定事件中处理指标数据文件
                                              scenario_metric_paths=[f"{output_dir}/{metric_dir}"],
                                              delete_scenario_metric_files=True)
    metric_summary_callback = MetricSummaryCallback(metric_save_path=f"{output_dir}/{metric_dir}",  # 创建指标汇总回调，用于在特定事件中处理指标数据汇总
                                                    metric_aggregator_save_path=f"{output_dir}/{aggregator_metric_dir}",
                                                    summary_output_path=f"{output_dir}/summary",
                                                    num_bins=20, pdf_file_name='summary.pdf')
    main_callbacks = MultiMainCallback([metric_file_callback, metric_aggregator_callback, metric_summary_callback]) # 创建主回调，用于在特定事件中处理主数据处理
    main_callbacks.on_run_simulation_start()    # 在模拟运行开始时调用主回调的启动方法，初始化回调过程

    # build simulation folder
    build_simulation_experiment_folder(output_dir, simulation_dir, metric_dir, aggregator_metric_dir)

    # build scenarios
    print('Extracting scenarios...')
    map_version = "nuplan-maps-v1.0"    # 定义地图的版本
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)   # 创建ScenarioMapping对象，用于映射场景，参数包括场景地图和采样比例
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version, scenario_mapping=scenario_mapping)  # 创建NuPlanScenarioBuilder对象，用于构建场景，参数包括数据路径、地图路径、地图版本和场景映射
    
    if args.load_test_set:  # 根据参数决定使用测试集还是根据给定参数过滤场景
        params = yaml.safe_load(open('test_scenario.yaml', 'r'))     # 如果使用测试集，从文件中加载测试场景参数
        scenario_filter = ScenarioFilter(**params)  # 根据加载的参数创建ScenarioFilter对象
    else:   # 如果不使用测试集，根据给定的场景数量参数获取过滤参数并创建ScenarioFilter对象
        scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type))
    
    worker = SingleMachineParallelExecutor(use_process_pool=False)  # 创建SingleMachineParallelExecutor对象，用于执行并行任务，参数表示是否使用进程池
    scenarios = builder.get_scenarios(scenario_filter, worker)  # 使用场景构建器、场景过滤器和并行执行器获取场景列表

    # begin testing
    print('Running simulations...')
    build_simulation(experiment_name, planner, scenarios, output_dir, simulation_dir, metric_dir)
    main_callbacks.on_run_simulation_end()
    simulation_file = [str(file) for file in pathlib.Path(output_dir).iterdir() if file.is_file() and file.suffix == '.nuboard']

    # show metrics and scenarios
    build_nuboard(builder, simulation_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)    # 添加数据路径参数，用于指定数据集的位置
    parser.add_argument('--map_path', type=str) # 添加地图路径参数，用于指定地图信息的位置
    parser.add_argument('--model_path', type=str)   # 添加模型路径参数，用于指定模型保存或加载的位置
    parser.add_argument('--test_type', type=str, default='closed_loop_nonreactive_agents')  # 添加测试类型参数，用于指定测试类型
    parser.add_argument('--load_test_set', action='store_true') # 添加是否加载测试集参数，用于指定是否加载测试集
    parser.add_argument('--device', type=str, default='cuda')   # 添加设备参数，用于指定模型运行的设备
    parser.add_argument('--scenarios_per_type', type=int, default=20)   # 添加测试场景数量参数，用于指定每种测试类型的测试场景数量
    args = parser.parse_args()

    main(args)
