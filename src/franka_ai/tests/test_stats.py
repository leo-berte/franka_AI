from pathlib import Path
import argparse
import numpy as np

from lerobot.common.datasets.compute_stats import aggregate_stats

from franka_ai.dataset.utils import load_episodes_stats_patch, LeRobotDatasetPatch


"""
Run the code: 

python src/franka_ai/tests/test_stats.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/today_data/today_outliers
python src/franka_ai/tests/test_stats.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/single/single_outliers

python src/franka_ai/tests/test_stats.py --dataset /workspace/data/today_data/today_outliers
"""


# TODO:

# 1) They don't match if I use: episodes_idx = [0,1]



def get_dataset_path():

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Absolute path to the dataset folder"
    )
    args = parser.parse_args()

    # return absolute path to the dataset
    return args.dataset

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
    dataset_path = get_dataset_path()

    # Set episodes indeces to load
    episodes_idx = [0,1]

    ## 1) Transformed stats

    # Get episodes stats
    episode_stats_path = Path(dataset_path) / "meta" / "episodes_stats_transformed.jsonl"
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
    check_stats(dataset_with_original_stats.meta.stats, new_dataset_stats)


if __name__ == "__main__":
    main()