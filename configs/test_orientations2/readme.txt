
# quaternioni

python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_act_2026-01-02_11-46-39 \
                                           --policy act
                                           
 # aa
 
python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_act_2026-01-02_14-21-38 \
                                           --policy act
 
 
 # 6D
 
python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_act_2026-01-02_14-47-55 \
                                           --policy act
         
         
         
         
                                           
# relative                                           
python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_act_2026-01-02_16-00-20 \
                                           --policy act                                           

# relative t2                                           
python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_act_2026-01-05_10-54-56 \
                                           --policy act    

# relative t4 8k steps                                          
python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_act_2026-01-05_12-06-18 \
                                           --policy act 

# relative t45 4k steps e lr ridotto di 10 factor                                         
python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_act_2026-01-05_15-04-37 \
                                           --policy act 
                                                                                     
# relative t3 no gripper                                          
python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_act_2026-01-05_11-22-53 \
                                           --policy act  
                                           
                                           
# relative: gripper to gripper only
python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_act_2026-01-05_15-44-24 \
                                           --policy act 

# relative: q, ee_rel, gripper to ee_rel , gripper
python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_act_2026-01-05_16-22-02 \
                                           --policy act                                            
                                           
# relative: ee_abs, ee_rel, gripper to ee_rel , gripper
python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_act_2026-01-05_16-22-02 \
                                           --policy act                                            
                                           
                                                                                                                                 
                                       
