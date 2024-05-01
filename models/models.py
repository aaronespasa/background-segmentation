# Encoders (segformer): https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/encoders/mix_transformer.py
# Decoder: https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/fpn/model.py
import segmentation_models_pytorch as smp

ENCODER_LIST = [f"mit_b{i}" for i in range(6)]
DECODERS = {"fpn": smp.FPN, "unet": smp.Unet}

def get_segformer_model(segformer_size, decoder, encoder_weights_dataset="imagenet", num_classes=1, in_channels=3):
    """
    Get Segformer model with specified encoder and decoder.
    Args:
        - segformer_size (str): Segformer size. Must be one of ['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'].
        - decoder (str): Decoder type. Must be one of ['fpn', 'unet'].
    """
    if segformer_size not in ENCODER_LIST:
        raise ValueError(f"Invalid segformer size: {segformer_size}. Must be one of {ENCODER_LIST}.")
    
    if decoder not in list(DECODERS.keys()):
        raise ValueError(f"Invalid decoder: {decoder}. Must be one of {list(DECODERS.keys())}.")
    
    model_class = DECODERS.get(decoder)

    if model_class is None:
        raise ValueError(f"Unsupported decoder: {decoder}.")

    model = model_class(
        encoder_name=segformer_size,
        encoder_weights=encoder_weights_dataset,
        in_channels=in_channels,
        classes=num_classes
    )

    smp.encoders.get_preprocessing_fn(encoder_name=segformer_size, pretrained=encoder_weights_dataset)
    
    return model