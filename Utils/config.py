import argparse

def parse_arguments():

    
    args = argparse.Namespace(
        debug=True,
        use_processed_pcd=False,
        instruction=None,
        scene_xml_path="Environment/scene.xml",
        pcd_path="Environment/concept_graph_results/pcd.pkl",
        processed_pcd_path="Environment/concept_graph_results/processed_pcd.pkl",
        objects_path="Environment/concept_graph_results/objects.json",
        relations_path="Environment/concept_graph_results/relations.json",
        grid_size=100,
        gpt_model="gpt-4o",
        trajectory_length=0.2,
        trajectory_steps=10,
        trajectory_radius=0.08,
        trajectory_obstacles_check=True,
        trajectory_tolerance=0.01,
        frequency=30,
        rerun=False,
        visualize_path=False,
        no_camera_visualization=True,
        disable_any_grasp=False,
        grasp_server_host="localhost",
        grasp_server_port=9875,
        min_distance=0.5
    )

    return args

    parser = argparse.ArgumentParser(description='Run UntidyBotSimulator with optional flags.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--use_processed_pcd', action='store_true', help='If True, use the processed PCD file')
    parser.add_argument('--instruction', type=str, help='Instruction to execute')
    parser.add_argument('--scene_xml_path', type=str, default="Environment/scene.xml", help='Mujoco XML scene path')
    parser.add_argument('--pcd_path', type=str, default="Environment/concept_graph_results/pcd.pkl", help='PCD pickle file path')
    parser.add_argument('--processed_pcd_path', type=str, default="Environment/concept_graph_results/processed_pcd.pkl", help='Processed PCD file path')
    parser.add_argument('--objects_path', type=str, default="Environment/concept_graph_results/objects.json", help='Objects JSON file path')
    parser.add_argument('--relations_path', type=str, default="Environment/concept_graph_results/relations.json", help='Relations JSON file path')
    parser.add_argument('--grid_size', type=int, default=100, help='Grid size for path planning')
    parser.add_argument('--gpt_model', type=str, default="gpt-4o", help='GPT model version')
    parser.add_argument('--trajectory_length', type=float, default=0.2, help='Length of trajectory for pick tasks')
    parser.add_argument('--trajectory_steps', type=int, default=10, help='Number of steps for the trajectory in pick tasks')
    parser.add_argument('--trajectory_radius', type=float, default=0.08, help='Radius for obstacle checking in trajectory')
    parser.add_argument('--trajectory_obstacles_check', type=bool, default=True, help='Enable obstacle checking in trajectory')
    parser.add_argument('--trajectory_tolerance', type=float, default=0.01, help='Tolerance for obstacle collision')
    parser.add_argument('--frequency', type=int, default=30, help='Simulation loop frequency')
    parser.add_argument('--rerun', action='store_true', help='Enable rerun.io visualization')
    parser.add_argument('--visualize_path', action='store_true', help='Visualize planned path')
    parser.add_argument('--no_camera_visualization', action='store_true', help='Disable camera visualization')
    parser.add_argument('--disable_any_grasp', action='store_true', help='Disable AnyGrasp Usage')
    parser.add_argument('--grasp_server_host', type=str, default="localhost", help='Grasping Server IP')
    parser.add_argument('--grasp_server_port', type=int, default=9875, help='Grasping Server Host')
    parser.add_argument('--min_distance', type=float, default=0.6, help='Minimum vaible distance from an object.')

    args = parser.parse_args()
    if args.debug:
        print_arguments(args) 
    return args

def print_arguments(args):
    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
