import segmentation_models_pytorch as smp

def get_model(model_name):
    models = {
        'UnetPlusPlus': smp.UnetPlusPlus,
        'Unet': smp.Unet,
        'FPN': smp.FPN,
        'Linknet': smp.Linknet,
        'MAnet': smp.MAnet,
        'PAN': smp.PAN,
        'PSPNet': smp.PSPNet,
        'DeepLabV3': smp.DeepLabV3,
        'DeepLabV3Plus': smp.DeepLabV3Plus,
        'UPerNet': smp.UPerNet
    }

    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name]

