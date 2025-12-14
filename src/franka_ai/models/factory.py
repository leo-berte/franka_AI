

def get_policy_class(policy_name):

    if policy_name == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        return DiffusionPolicy
    elif policy_name == "act":
        # from lerobot.common.policies.act.modeling_act import ACTPolicy
        from franka_ai.models.actPatch.modeling_act import ACTPolicy
        return ACTPolicy
    else:
        raise NotImplementedError(f"Policy with name {policy_name} is not implemented.")

def get_policy_config_class(policy_name):

    if policy_name == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionConfig
        return DiffusionConfig
    elif policy_name == "act":
        # from lerobot.common.policies.act.configuration_act import ACTConfig
        from franka_ai.models.actPatch.configuration_act import ACTConfig
        return ACTConfig
    else:
        raise NotImplementedError(f"Policy with name {policy_name} is not implemented.")

def make_policy(policy_name, config, dataset_stats=None):

    PolicyClass = get_policy_class(policy_name)

    if dataset_stats is not None:
        return PolicyClass(config, dataset_stats=dataset_stats)
    
    return PolicyClass(config)