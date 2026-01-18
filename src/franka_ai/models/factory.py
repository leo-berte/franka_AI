# Factory utils to upload policy models

def get_policy_class(policy_name):

    if policy_name == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        return DiffusionPolicy
    elif policy_name == "act":
        from franka_ai.models.actPatch.modeling_act import ACTPolicy
        return ACTPolicy
    elif policy_name == "flowLeonardo":
        from franka_ai.models.flowLeonardo.modeling_flow import FlowPolicy
        return FlowPolicy
    elif policy_name == "flowMathis":
        from franka_ai.models.flowMathis.modeling_flow import FlowPolicy
        return FlowPolicy
    elif policy_name == "template":
        from franka_ai.models.template.modeling_template import TemplatePolicy
        return TemplatePolicy
    else:
        raise NotImplementedError(f"Policy with name {policy_name} is not implemented.")

def get_policy_config_class(policy_name):

    if policy_name == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionConfig
        return DiffusionConfig
    elif policy_name == "act":
        from franka_ai.models.actPatch.configuration_act import ACTConfig
        return ACTConfig
    elif policy_name == "flowLeonardo":
        from franka_ai.models.flowLeonardo.configuration_flow import FlowConfig
        return FlowConfig
    elif policy_name == "flowMathis":
        from franka_ai.models.flow.configuration_flow import FlowConfig
        return FlowConfig
    elif policy_name == "template":
        from franka_ai.models.template.configuration_template import TemplateConfig
        return TemplateConfig
    else:
        raise NotImplementedError(f"Policy with name {policy_name} is not implemented.")

def make_policy(policy_name, config, dataset_stats=None):

    PolicyClass = get_policy_class(policy_name)

    if dataset_stats is not None:
        return PolicyClass(config, dataset_stats=dataset_stats)
    
    return PolicyClass(config)