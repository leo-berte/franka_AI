

def get_policy_class(policy_name):

    if policy_name == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        return DiffusionPolicy
    elif policy_name == "act":
        from franka_ai.models.actPatch.modeling_act_original import ACTPolicy
        return ACTPolicy
    elif policy_name == "act_mathis":
        from franka_ai.models.actPatch.modeling_act_mathis import ACTPolicyPatch
        return ACTPolicyPatch
    elif policy_name == "flow":
        from franka_ai.models.flow.modeling_flow import FlowPolicy
        return FlowPolicy
    else:
        raise NotImplementedError(f"Policy with name {policy_name} is not implemented.")

def get_policy_config_class(policy_name):

    if policy_name == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionConfig
        return DiffusionConfig
    elif policy_name == "act":
        from franka_ai.models.actPatch.configuration_act_original import ACTConfig
        return ACTConfig
    elif policy_name == "act_mathis":
        from franka_ai.models.actPatch.configuration_act_mathis import ACTConfigPatch
        return ACTConfigPatch
    elif policy_name == "flow":
        from franka_ai.models.flow.configuration_flow import FlowConfig
        return FlowConfig
    else:
        raise NotImplementedError(f"Policy with name {policy_name} is not implemented.")

def make_policy(policy_name, config, dataset_stats=None):

    PolicyClass = get_policy_class(policy_name)

    if dataset_stats is not None:
        return PolicyClass(config, dataset_stats=dataset_stats)
    
    return PolicyClass(config)