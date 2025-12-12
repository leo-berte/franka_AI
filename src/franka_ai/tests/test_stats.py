from pathlib import Path
import argparse
import numpy as np

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from franka_ai.dataset.utils import load_episodes_stats_patch, LeRobotDatasetPatch


"""
Run the code: 

python src/franka_ai/tests/test_stats.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/single/single_outliers --config config1
python src/franka_ai/tests/test_stats.py --dataset /workspace/data/single_outliers --config config1
"""

def parse_args():

    # set parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
        type=str,
        required=True,
        help="Absolute path to the dataset folder")
        
    parser.add_argument("--config", type=str, default="config",
                help="Config folder name")
    
    args = parser.parse_args()

    # return absolute path to the dataset and config folder name
    return args.dataset, args.config

def check_stats(old_stats, new_stats):

    TOL = 1E-2

    # loop over features
    for feature_key,feature_stats in old_stats.items():
       
        # features not present in both stats are not considered 
        if feature_key not in new_stats.keys():
           continue
           
        # loop over min, max, mean, std, count
        for stats_type,values in feature_stats.items(): 
           
            # check values for each stat type
            for idx, val in enumerate(zip(values, new_stats[feature_key][stats_type])):

                # print indeces where there is a mismatch
                if (np.any(np.abs(val[0] - val[1]) > TOL)):
                    print(f"[ERROR]: Feature [{feature_key}] don't match for stat type [{stats_type}] at index {idx} --> {val}")

def main():

    # Get path to dataset (via argparser)
    dataset_path, config_folder = parse_args()

    # Get dataset metadata
    dataset_meta = LeRobotDatasetMetadata(repo_id=None, root=dataset_path)
    num_episodes = dataset_meta.total_episodes

    # Set episodes indeces to load
    episodes_idx = list(range(20))
    # episodes_idx = list(range(num_episodes))

    ## 1) Transformed stats

    # Get episodes stats
    episode_stats_path = Path(f"configs/{config_folder}") / "episodes_stats_transformed.jsonl"
    episodes_stats = load_episodes_stats_patch(episode_stats_path)

    # Aggregate episodes stats in a unique global stats
    new_dataset_stats = aggregate_stats([episodes_stats[ep] for ep in episodes_idx])
    
    # print("Transformed stats: ")
    # print(new_dataset_stats["index"])
    # print(new_dataset_stats["observation.state"])
    
    ## 2) Original stats

    # Load the raw dataset
    dataset_with_original_stats = LeRobotDatasetPatch(
        repo_id=None,
        root=dataset_path,   
        episodes=episodes_idx
    )

    # print("Original stats: ")
    # print(dataset_with_original_stats.meta.stats["index"])
    # print(dataset_with_original_stats.meta.stats["observation.state"])

    # Compare stats
    check_stats(dataset_with_original_stats.stats, new_dataset_stats)


if __name__ == "__main__":
    main()