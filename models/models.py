# Encoders (segformer): https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/encoders/mix_transformer.py
# Decoder: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/fpn/model.py
import segmentation_models_pytorch as smp

ENCODER_LIST = [f"mit_b{i}" for i in range(6)] + ["resnet50", "mobileone_s4"]
DECODERS = {"fpn": smp.FPN, "unet": smp.Unet, "manet": smp.MAnet, "deeplabv3": smp.DeepLabV3}

def get_model(encoder, decoder, encoder_weights_dataset="imagenet", num_classes=1, in_channels=3):
    """
    Get Segformer model with specified encoder and decoder.
    Args:
        - encoder (str): Encoder name. Must be one of ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5', 'resnet50', 'mobileone_s4', 'efficientnet-b5'].
        - decoder (str): Decoder type. Must be one of ['fpn', 'unet', 'manet', 'deeplabv3'].
    """
    if encoder not in ENCODER_LIST:
        raise ValueError(f"Invalid segformer size: {encoder}. Must be one of {ENCODER_LIST}.")
    
    if decoder not in list(DECODERS.keys()):
        raise ValueError(f"Invalid decoder: {decoder}. Must be one of {list(DECODERS.keys())}.")

    model_class = DECODERS.get(decoder)

    if model_class is None:
        raise ValueError(f"Unsupported decoder: {decoder}.")

    model = model_class(
        encoder_name=encoder,
        encoder_weights=encoder_weights_dataset,
        in_channels=in_channels,
        classes=num_classes
    )

    smp.encoders.get_preprocessing_fn(encoder_name=encoder, pretrained=encoder_weights_dataset)
    
    return model


