import warnings

from torch import nn

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import types

import torch
import torch.cuda
from flwr.client import NumPyClient, Client
from flwr.common import Context
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer
from opacus.utils.batch_memory_manager import wrap_data_loader
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import DataLoader

from src.picai_baseline.flwr.dp_state_manager import DPStateManager
from src.picai_baseline.unet.training_setup.data_generator import DataLoaderFromDataset, default_collate
from src.picai_baseline.flwr.federated_training_methods import load_datasets, set_parameters, \
    get_parameters, train, test
from src.picai_baseline.flwr.run_config import run_configuration
from src.picai_baseline.unet.training_setup.augmentations.nnUNet_DA import apply_augmentations
from src.picai_baseline.unet.training_setup.compute_spec import compute_spec_for_run
from src.picai_baseline.unet.training_setup.loss_functions.focal import FocalLoss
from src.picai_baseline.unet.training_setup.neural_network_selector import neural_network_for_run
from opacus import PrivacyEngine
import gc

from pathlib import Path

import opacus.grad_sample.utils

print(opacus.grad_sample.utils.__file__)


class PicaiFlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, optimizer, loss_func, arguments,
                 device, partition_id, privacy_engine: PrivacyEngine, dp_state_manager):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.args = arguments
        self.device = device
        self.partition_id = partition_id
        self.privacy_engine = privacy_engine
        self.dp_state_manager: DPStateManager = dp_state_manager

    def get_parameters(self, config):
        return get_parameters(self.net)

    def get_length_of_data(self, loader):
        dataset_length = 0
        if hasattr(loader, 'generator') and hasattr(loader.generator, 'get_data_length'):
            dataset_length = loader.generator.get_data_length()
        elif hasattr(loader, 'dataset') and hasattr(loader.dataset, '__len__'):
            dataset_length = len(loader.dataset)
        elif hasattr(loader, '_data') and hasattr(loader._data, '__len__'):
            dataset_length = len(loader._data)
        else:
            # Fallback to a reasonable value based on your dataset
            dataset_length = 1200  # Approximate based on your logs
        return dataset_length

    def fit(self, parameters, config):
        round_idx = config.get("current_round", 0)
        print(f"Partition id {self.partition_id}")
        eps_before, _ = self.privacy_engine.accountant.get_privacy_spent(delta=self.args.delta)
        print(f"[Client {self.partition_id}] ▷ Before this round, cumulative ε = {eps_before:.4f}")
        set_parameters(self.net, parameters)
        train_loss, all_norms = train(self.net, self.optimizer, self.loss_func, self.trainloader, self.args, self.device,
                             config["local_epochs"])

        # store current norms
        if all_norms is not None and all_norms.numel() > 0:
            median_norm = all_norms.median().item()
            p90_norm = torch.quantile(all_norms, 0.9).item()
            max_norm = torch.max(all_norms).item()
            mean_norm = torch.mean(all_norms).item()
            min_norm = torch.min(all_norms).item()
            self.dp_state_manager.log_grad_norms(
                self.partition_id, round_idx, median_norm, p90_norm, mean_norm, min_norm, max_norm
            )

        # 3) Query the Opacus accountant for spent ε (and α)
        #    Make sure to use the same δ you chose in your RunConfig
        eps_after, alpha = self.privacy_engine.accountant.get_privacy_spent(
            delta=self.args.delta
        )
        print(f"[Client {self.partition_id}] ◁ After this round, cumulative ε = {eps_after:.4f}, α = {alpha}")

        self.dp_state_manager.save(self.partition_id, self.privacy_engine.accountant)

        # 4) Return updated weights, num examples, and ε as a metric
        #    You can also include alpha if you want:
        metrics = {"epsilon": float(eps_after), "alpha": float(alpha), 'train_loss': train_loss}

        return get_parameters(self.net), self.get_length_of_data(self.trainloader), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        tracking_metrics = test(self.net, self.optimizer, self.loss_func, self.valloader, self.args,
                                self.device)
        loss = float(tracking_metrics["loss"])
        tracking_metrics.pop('loss')

        return loss, self.get_length_of_data(self.valloader), tracking_metrics


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""
    gc.collect()
    torch.cuda.empty_cache()
    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]

    dp_state_manager = DPStateManager()

    # Check if we already have a saved accountant
    privacy_engine = PrivacyEngine(accountant="rdp")

    fold_id = partition_id % len(run_configuration.folds)

    # Call compute_spec_for_run with assigned GPU
    device = compute_spec_for_run()

    print(f"Initial Device type: {type(device)}, Value: {device}")

    trainloader, valloader, class_weights = load_datasets(fold_id=fold_id)

    wrapped_dataset = PicaiDatasetWrapper(trainloader)
    print(f"Client {partition_id} dataset length = {len(wrapped_dataset)}")

    train_data_loader = DataLoader(wrapped_dataset, batch_size=run_configuration.virtual_batch_size, shuffle=True,
                                   collate_fn=default_collate, num_workers=0, pin_memory=False)

    # Load model
    net = neural_network_for_run(args=run_configuration, device=device)

    def freeze_encoder_and_bottleneck(monai_unet: nn.Module):
        """
        Freeze every submodule in monai_unet.model whose `conv` is a bare Conv3d and either:
          • has stride > 1  (down-sampling / encoder),  OR
          • has out_channels == 1024  (the bottleneck layer in this UNet).
        All other submodules (the up-sampling / decoder side) remain trainable.
        """
        for module in monai_unet.model.modules():
            # 1) Does this block have an attribute `conv` that is a bare Conv3d?
            if hasattr(module, "conv") and isinstance(module.conv, nn.Conv3d):
                conv3d = module.conv
                # 2a) If any dimension of conv3d.stride > 1, it's an encoder/down-sampler
                is_down = any(s > 1 for s in conv3d.stride)
                # 2b) If conv3d.out_channels == 1024, assume it's the bottleneck
                is_bottleneck = (conv3d.out_channels == 1024)
                if is_down or is_bottleneck:
                    for p in module.parameters():
                        p.requires_grad = False


    #freeze_encoder_and_bottleneck(net)

    for name, param in net.named_parameters():
        print(name, param.requires_grad, tuple(param.shape))

    # Count how many parameters remain trainable:
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    # loss function + optimizer
    loss_func = FocalLoss(
        alpha=class_weights[-1],
        gamma=run_configuration.focal_loss_gamma).to(device)
    trainable = filter(lambda p: p.requires_grad, net.parameters())

    optimizer = torch.optim.Adam(params=trainable, lr=run_configuration.base_lr, amsgrad=True, weight_decay=1e-4)

    current_epsilon = run_configuration.epsilon
    if dp_state_manager.exists(partition_id):
        restored = dp_state_manager.load(partition_id)
        privacy_engine.accountant = restored
        print(f"[Client {partition_id}] ✅ Accountant fully restored with {len(restored)} steps")
        current_epsilon = current_epsilon - privacy_engine.accountant.get_epsilon(delta=run_configuration.delta)

    #adaptive_clip = dp_state_manager.get_adaptive_clip_value(partition_id, factor=1.0)
    adaptive_clip = None
    if adaptive_clip is not None:
        run_configuration.max_grad_norm = adaptive_clip  # override before make_private_with_epsilon
    else:
        adaptive_clip = run_configuration.max_grad_norm

    print(f"Adaptive clip value is {adaptive_clip}")

    net, private_optimizer, private_train_loader = privacy_engine.make_private_with_epsilon(module=net,
                                                                                            optimizer=optimizer,
                                                                                            data_loader=train_data_loader,
                                                                                            target_epsilon=current_epsilon,
                                                                                            target_delta=run_configuration.delta,
                                                                                            epochs=run_configuration.num_train_epochs * run_configuration.num_rounds,
                                                                                            max_grad_norm=adaptive_clip,
                                                                                            grad_sample_mode="functorch"
                                                                                            )

    # 1) Optimizer is wrapped
    assert isinstance(private_optimizer, DPOptimizer), "Optimizer isn’t a DPOptimizer!"

    # 2) Sampler is Poisson (Uniform with replacement)
    bs = private_train_loader.batch_sampler
    assert isinstance(bs, UniformWithReplacementSampler), f"Sampler is {type(bs)}"

    # 3) Accountant is RDPAccountant
    assert isinstance(privacy_engine.accountant,
                      RDPAccountant), f"Privacy engine accountant is {type(privacy_engine.accountant)}"
    print("✅ DPOptimizer, Poisson sampler, and RDP accountant are in place.")

    private_train_loader = wrap_data_loader(
        data_loader=private_train_loader,
        max_batch_size=run_configuration.physical_batch_size,
        optimizer=private_optimizer,
    )

    private_train_loader.set_thread_id = types.MethodType(set_thread_id, private_train_loader)
    private_train_loader.pin_memory = False
    private_train_loader.pin_memory_queue = None

    train_gen = apply_augmentations(
        dataloader=private_train_loader,
        num_threads=run_configuration.num_threads_augmenting,
        disable=(not bool(run_configuration.enable_da)),
    )

    # initialize multi-threaded augmenter in background
    if hasattr(train_gen, "restart"):
        train_gen.restart()

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return PicaiFlowerClient(net, train_gen, valloader, private_optimizer, loss_func, run_configuration, device,
                             partition_id, privacy_engine, dp_state_manager).to_client()


class PicaiDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, original_loader: DataLoaderFromDataset):
        self.original_loader = original_loader
        # Determine dataset length if available, otherwise use a reasonable default
        self.length = self.original_loader.get_data_length()
        self._data = original_loader.dataset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_item, seg_item = self._data[idx]

        # Return in the format expected by the model
        return {
            'data': data_item.numpy(),
            'seg': seg_item.numpy()
        }


def set_thread_id(self, thread_id):
    self.thread_id = thread_id


def enforce_dict_batches(train_gen):
    """Wrap any generator of [ (data, seg) ] or {"data","seg"} into only dicts."""
    for batch in train_gen:
        # Already a dict? Just pass it through
        if isinstance(batch, dict):
            yield batch
            continue

        # Otherwise expect a tuple/list of length 2
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            yield {"data": batch[0], "seg": batch[1]}
            continue

        # Anything else is a bug
        raise RuntimeError(f"Unexpected batch format in train_gen: {type(batch)}")


def get_client_dp_state_path(participant_id: int) -> Path:
    path = Path.home() / ".flwr_dp_states"
    path.mkdir(exist_ok=True)
    return path / f"client_{participant_id}.pt"
