import os
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra.utils import instantiate
from omegaconf import OmegaConf

from tsd.models.base.encoder_model_base import EncoderModelBase
from tsd.tasks.dmcontrol_task import DMControlTask


class SymbolicEncoder(nn.Module):
    def __init__(self, observation_size, embedding_size, activation_function="relu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)

    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden


class VisualEncoder(nn.Module):
    def __init__(self, embedding_size, activation_function="relu"):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = (
            nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        )  # identity if embedding size is 1024 else fully connected layer

    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)
        return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function="relu"):
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        return VisualEncoder(embedding_size, activation_function)

class TSDAgentEncoder(nn.Module):

    def __init__(self, tsd_encoder_configuration_name, repo_config):
        super().__init__()

        # Create the task config needed to assemble the TSD encoder
        domain, task_name = repo_config.env_id.split("-", 1)
        if domain == "dmc":
            task_config = OmegaConf.create(
                {
                    "cfg": {
                        "task_name": task_name,
                        "training_data_path": f"{repo_config.offline_dir}/tdmpc2/mt30-{task_name}-training.pt",
                        "validation_data_path": f"{repo_config.offline_dir}/tdmpc2/mt30-{task_name}-validation.pt",
                        "seed": 1,
                        "horizon": 4,
                        "fraction": 1.0,
                        "overlap_ratio": 0.0,
                        "load_device": f"cuda:{repo_config.gpu_id}",
                        "batch_device": f"cuda:{repo_config.gpu_id}",
                    }
                }
            )
        # Instantiate the encoder
        config_path = os.path.join(os.path.dirname(__file__), f"{tsd_encoder_configuration_name}.yaml")
        tsd_encoder_configuration = OmegaConf.load(config_path)
        # Set the relevant config variables
        tsd_encoder_configuration.task = task_config
        tsd_encoder_configuration.cfg.checkpoint_dir = repo_config.checkpoint_dir
        tsd_encoder_configuration.cfg.data_dir = repo_config.offline_dir
        tsd_encoder_configuration.cfg.device = f"cuda:{repo_config.gpu_id}"
        with patch("hydra.utils.get_original_cwd", return_value=os.getcwd()):
            self.tsd_encoder = instantiate(tsd_encoder_configuration)
        self.tsd_encoder.requires_grad_(False)

    def requires_grad_(self, requires_grad = True):
        return super().requires_grad_(requires_grad and self.tsd_configuration.model.cfg.freeze)

    def forward(self, observation):
        return self.tsd_encoder.encode(observation)


class ConditionalSymbolicEncoder(SymbolicEncoder):
    def __init__(
        self,
        observation_size,
        embedding_size,
        condition_size,
        activation_function="relu",
    ):
        super().__init__(
            observation_size + condition_size, embedding_size, activation_function
        )

    def forward(self, observation, condition):
        pseudo_observation = torch.cat((observation, condition), dim=1)
        return super().forward(pseudo_observation)


class ConditionalVisualEncoder(VisualEncoder):
    def __init__(self, embedding_size, condition_size, activation_function="relu"):
        super().__init__(embedding_size, activation_function)
        self.condition_size = condition_size
        self.film = nn.Linear(condition_size, 2 * (32 + 64 + 128 + 256))

    def mod(self, x, gamma, beta):
        return (1 + gamma[..., None, None]) * x + beta[..., None, None]

    def forward(self, observation, condition):
        gammas, betas = self.film(condition).chunk(2, dim=1)
        gammas = gammas.split([32, 64, 128, 256], dim=1)
        betas = betas.split([32, 64, 128, 256], dim=1)

        hidden = self.act_fn(self.mod(self.conv1(observation), gammas[0], betas[0]))
        hidden = self.act_fn(self.mod(self.conv2(hidden), gammas[1], betas[1]))
        hidden = self.act_fn(self.mod(self.conv3(hidden), gammas[2], betas[2]))
        hidden = self.act_fn(self.mod(self.conv4(hidden), gammas[3], betas[3]))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)
        return hidden


def ConditionalEncoder(
    symbolic,
    observation_size,
    embedding_size,
    condition_size,
    activation_function="relu",
):
    if symbolic:
        return ConditionalSymbolicEncoder(
            observation_size, embedding_size, condition_size, activation_function
        )
    else:
        return ConditionalVisualEncoder(
            embedding_size, condition_size, activation_function
        )


class DummyConditionalSymbolicEncoder(SymbolicEncoder):
    def __init__(
        self,
        observation_size,
        embedding_size,
        condition_size,
        activation_function="relu",
    ):
        super().__init__(observation_size, embedding_size, activation_function)

    def forward(self, observation, condition):
        return super().forward(observation)


class DummyConditionalVisualEncoder(VisualEncoder):
    def __init__(self, embedding_size, condition_size, activation_function="relu"):
        super().__init__(embedding_size, activation_function)

    def forward(self, observation, condition):
        return super().forward(observation)


def DummyConditionalEncoder(
    symbolic,
    observation_size,
    embedding_size,
    condition_size,
    activation_function="relu",
):
    if symbolic:
        return DummyConditionalSymbolicEncoder(
            observation_size, embedding_size, condition_size, activation_function
        )
    else:
        return DummyConditionalVisualEncoder(
            embedding_size, condition_size, activation_function
        )
