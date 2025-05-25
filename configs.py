from pathlib import Path
from omegaconf import DictConfig

from datamodule import DataModule
from add_thin.diffusion.model import AddThin
from add_thin.backbones.classifier import PointClassifier
from add_thin.distributions.intensities import MixtureIntensity
from tasks import DensityEstimation
from discrete_diffusion.diffusion_transformer import DiffusionTransformer

def instantiate_datamodule(config: DictConfig):
    return DataModule(
        config.root,
        config.name,
        batch_size=config.batch_size,
    )

def instantiate_model(config: DictConfig, datamodule) -> AddThin:
    classifier = PointClassifier(
        hidden_dims=config.temporal_hidden_dims,
        layer=config.temporal_classifier_layer,
    )
    intensity = MixtureIntensity(
        n_components=config.temporal_mix_components,
        embedding_size=config.temporal_hidden_dims,
        distribution="normal",
        time_segments=config.temporal_time_segments,
    )

    tpp_model = AddThin(
        classifier_model=classifier,
        intensity_model=intensity,
        max_time=datamodule.train_data.tmax.item(),
        steps=config.temporal_steps,
        hidden_dims=config.temporal_hidden_dims,
        emb_dim=config.temporal_hidden_dims,
        encoder_layer=config.temporal_encoder_layer,
        n_max=datamodule.n_max,
        kernel_size=config.temporal_kernel_size,
        num_condition_types=config.num_condition_types,
        time_segments=config.temporal_time_segments,
    )

    discrete_diffusion =  DiffusionTransformer(
        diffusion_step=config.spatial_hidden_dims,
        alpha_init_type='alpha1',
        type_classes=datamodule.num_category,
        poi_classes=datamodule.num_poi,
        num_condition_types=config.num_condition_types,
    )
    return tpp_model, discrete_diffusion


def instantiate_task(config: DictConfig, tpp_model, discrete_diffusion):
    return DensityEstimation(
        tpp_model,
        discrete_diffusion,
        config.learning_rate1,
        config.learning_rate2,
        config.weight_decay1,
        config.weight_decay2,
    )

