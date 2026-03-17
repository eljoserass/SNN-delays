from utils import set_seed

import os
import shutil
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from typing import Callable, Optional

import torchvision.transforms as transforms

from spikingjelly.datasets.shd import SpikingHeidelbergDigits
from spikingjelly.datasets import pad_sequence_collate
try:
    from spikingjelly.datasets.shd import SpikingSpeechCommands
except ImportError:
    SpikingSpeechCommands = None

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import augmentations
try:
    import torchaudio
    from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB, Resample
    from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
except ImportError:
    torchaudio = None
    Spectrogram = None
    MelScale = None
    AmplitudeToDB = None
    Resample = None
    SPEECHCOMMANDS = None


class RNoise(object):
  
  def __init__(self, sig):
    self.sig = sig
        
  def __call__(self, sample):
    noise = np.abs(np.random.normal(0, self.sig, size=sample.shape).round())
    return sample + noise


class TimeNeurons_mask_aug(object):

  def __init__(self, config):
    self.config = config
  
  
  def __call__(self, x, y):
    # Sample shape: (time, neurons)
    for sample in x:
      # Time mask
      if np.random.uniform() < self.config.TN_mask_aug_proba:
        mask_size = np.random.randint(0, self.config.time_mask_size)
        ind = np.random.randint(0, sample.shape[0] - self.config.time_mask_size)
        sample[ind:ind+mask_size, :] = 0

      # Neuron mask
      if np.random.uniform() < self.config.TN_mask_aug_proba:
        mask_size = np.random.randint(0, self.config.neuron_mask_size)
        ind = np.random.randint(0, sample.shape[1] - self.config.neuron_mask_size)
        sample[:, ind:ind+mask_size] = 0

    return x, y


class CutMix(object):
  """
  Apply Spectrogram-CutMix augmentaiton which only cuts patch across time axis unlike 
  typical Computer-Vision CutMix. Applies CutMix to one batch and its shifted version.
    
  """

  def __init__(self, config):
    self.config = config
  
  
  def __call__(self, x, y):
    
    # x shape: (batch, time, neurons)
    # Go to L-1, no need to augment last sample in batch (for ease of coding)

    for i in range(x.shape[0]-1):
      # other sample to cut from
      j = i+1
      
      if np.random.uniform() < self.config.cutmix_aug_proba:
        lam = np.random.uniform()
        cut_size = int(lam * x[j].shape[0])

        ind = np.random.randint(0, x[i].shape[0] - cut_size)

        x[i][ind:ind+cut_size, :] = x[j][ind:ind+cut_size, :]

        y[i] = (1-lam) * y[i] + lam * y[j]

    return x, y



class Augs(object):

  def __init__(self, config):
    self.config = config
    self.augs = [TimeNeurons_mask_aug(config), CutMix(config)]
  
  def __call__(self, x, y):
    for aug in self.augs:
      x, y = aug(x, y)
    
    return x, y



def _resolve_dataset_root(config, dataset_name):
  dataset_root = config.datasets_path
  base_name = os.path.basename(os.path.normpath(dataset_root)).lower()
  resolved = dataset_root
  if base_name == dataset_name.lower():
    resolved = dataset_root
  elif base_name == 'datasets':
    resolved = os.path.join(dataset_root, dataset_name.upper())
  return os.path.abspath(resolved)


def _contains_frame_files(root_dir):
  if not os.path.exists(root_dir):
    return False
  for _, _, files in os.walk(root_dir):
    for f in files:
      if f.endswith('.npz') or f.endswith('.npy'):
        return True
  return False


def _to_numpy_frames(frames, n_bins):
  if isinstance(frames, torch.Tensor):
    frames = frames.detach().cpu().numpy()
  frames = np.asarray(frames, dtype=np.float32)
  if frames.ndim != 2:
    raise ValueError(f'Expected 2D frames, got shape={frames.shape}')
  # Keep axis-1 as the binned axis whenever possible. If only axis-0 is compatible
  # with n_bins, transpose for older loader layouts.
  if frames.shape[1] % n_bins != 0 and frames.shape[0] % n_bins == 0:
    frames = frames.T
  return frames


def _bin_frames(frames, n_bins):
  binned_len = frames.shape[1] // n_bins
  binned_frames = np.zeros((frames.shape[0], binned_len), dtype=np.float32)
  for i in range(binned_len):
    binned_frames[:, i] = frames[:, n_bins * i: n_bins * (i + 1)].sum(axis=1)
  return binned_frames


def SHD_dataloaders(config):
  dataset_root = _resolve_dataset_root(config, 'shd')
  extract_root = os.path.join(dataset_root, 'extract')
  expected_h5 = ('shd_train.h5', 'shd_test.h5')
  os.makedirs(dataset_root, exist_ok=True)
  if os.path.exists(extract_root):
    missing_h5 = [f for f in expected_h5 if not os.path.exists(os.path.join(extract_root, f))]
    if missing_h5:
      print(f"Incomplete SHD extract found (missing: {missing_h5}). Rebuilding extract directory.")
      shutil.rmtree(extract_root)

  set_seed(config.seed)

  print(f"===> SHD data root = {dataset_root}")

  train_dataset = BinnedSpikingHeidelbergDigits(dataset_root, config.n_bins, train=True, data_type='frame', duration=config.time_step)
  test_dataset= BinnedSpikingHeidelbergDigits(dataset_root, config.n_bins, train=False, data_type='frame', duration=config.time_step)

  #train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2])

  train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
  #valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size)
  test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

  return train_loader, test_loader




def SSC_dataloaders(config):
  dataset_root = _resolve_dataset_root(config, 'ssc')
  set_seed(config.seed)

  os.makedirs(dataset_root, exist_ok=True)
  print(f"===> SSC data root = {dataset_root}")

  extract_root = os.path.join(dataset_root, 'extract')
  expected_h5 = ('ssc_train.h5', 'ssc_valid.h5', 'ssc_test.h5')
  if os.path.exists(extract_root):
    missing_h5 = [f for f in expected_h5 if not os.path.exists(os.path.join(extract_root, f))]
    if missing_h5:
      print(f"Incomplete SSC extract found (missing: {missing_h5}). Rebuilding extract directory.")
      shutil.rmtree(extract_root)

  events_root = os.path.join(dataset_root, 'events_h5')
  if os.path.exists(events_root):
    missing_events = [f for f in expected_h5 if not os.path.exists(os.path.join(events_root, f))]
    if missing_events:
      print(f"Incomplete SSC events_h5 found (missing: {missing_events}). Rebuilding events_h5 directory.")
      shutil.rmtree(events_root)

  frames_root = os.path.join(dataset_root, f'duration_{config.time_step}')
  expected_splits = ('train', 'valid', 'test')
  if os.path.exists(frames_root):
    missing_splits = [s for s in expected_splits if not os.path.isdir(os.path.join(frames_root, s))]
    empty_splits = [s for s in expected_splits if os.path.isdir(os.path.join(frames_root, s)) and not _contains_frame_files(os.path.join(frames_root, s))]
    if missing_splits or empty_splits:
      print(
        f"Incomplete SSC frames found in {frames_root}. "
        f"Missing splits: {missing_splits}, empty splits: {empty_splits}. Rebuilding frames directory."
      )
      shutil.rmtree(frames_root)

  train_dataset = BinnedSpikingSpeechCommands(dataset_root, config.n_bins, split='train', data_type='frame', duration=config.time_step)
  valid_dataset = BinnedSpikingSpeechCommands(dataset_root, config.n_bins, split='valid', data_type='frame', duration=config.time_step)
  test_dataset = BinnedSpikingSpeechCommands(dataset_root, config.n_bins, split='test', data_type='frame', duration=config.time_step)


  train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
  valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)
  test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

  return train_loader, valid_loader, test_loader

def GSC_dataloaders(config):
  if torchaudio is None:
    raise ImportError("torchaudio is required for GSC dataloaders.")
  dataset_root = _resolve_dataset_root(config, 'gsc')
  set_seed(config.seed)

  os.makedirs(dataset_root, exist_ok=True)
  print(f"===> GSC data root = {dataset_root}")

  train_dataset = GSpeechCommands(dataset_root, 'training', transform=build_transform(False), target_transform=target_transform)
  valid_dataset = GSpeechCommands(dataset_root, 'validation', transform=build_transform(False), target_transform=target_transform)
  test_dataset = GSpeechCommands(dataset_root, 'testing', transform=build_transform(False), target_transform=target_transform)


  train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
  valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=4)
  test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)

  return train_loader, valid_loader, test_loader


class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    def __init__(
            self,
            root: str,
            n_bins: int,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The Spiking Heidelberg Digits (SHD) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        """
        super().__init__(
            root=root,
            train=train,
            data_type=data_type,
            frames_number=frames_number,
            split_by=split_by,
            duration=duration,
            custom_integrate_function=custom_integrate_function,
            custom_integrated_frames_dir_name=custom_integrated_frames_dir_name,
            transform=transform,
            target_transform=target_transform,
        )
        self.n_bins = n_bins
        self.data_type = getattr(self, 'data_type', data_type)
        self.transform = getattr(self, 'transform', transform)
        self.target_transform = getattr(self, 'target_transform', target_transform)

    @classmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        # Use absolute targets to avoid relative-symlink breakage on some hosts.
        for f in Path(extract_root).iterdir():
            target = Path(raw_root) / f.name
            if target.exists():
                continue
            target.symlink_to(f.resolve())

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, label

        elif self.data_type == 'frame':
            local_fields_ok = hasattr(self, 'frames_path') and hasattr(self, 'frames_label')
            if local_fields_ok:
                frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
                label = self.frames_label[i]
                apply_local_transform = True
            else:
                # Newer SpikingJelly revisions changed internal frame bookkeeping.
                frames, label = super().__getitem__(i)
                apply_local_transform = False

            frames = _to_numpy_frames(frames, self.n_bins)
            binned_frames = _bin_frames(frames, self.n_bins)

            if apply_local_transform:
                if self.transform is not None:
                    binned_frames = self.transform(binned_frames)
                if self.target_transform is not None:
                    label = self.target_transform(label)

            return binned_frames, label



if SpikingSpeechCommands is not None:
    class BinnedSpikingSpeechCommands(SpikingSpeechCommands):
        def __init__(
                self,
                root: str,
                n_bins: int,
                split: str = 'train',
                data_type: str = 'event',
                frames_number: int = None,
                split_by: str = None,
                duration: int = None,
                custom_integrate_function: Callable = None,
                custom_integrated_frames_dir_name: str = None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
        ) -> None:
            """
            The Spiking Speech Commands (SSC) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

            Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

            .. admonition:: Note
                :class: note

                Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

            :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
            """
            super().__init__(
                root=root,
                split=split,
                data_type=data_type,
                frames_number=frames_number,
                split_by=split_by,
                duration=duration,
                custom_integrate_function=custom_integrate_function,
                custom_integrated_frames_dir_name=custom_integrated_frames_dir_name,
                transform=transform,
                target_transform=target_transform,
            )
            self.n_bins = n_bins
            self.data_type = getattr(self, 'data_type', data_type)
            self.transform = getattr(self, 'transform', transform)
            self.target_transform = getattr(self, 'target_transform', target_transform)

        @classmethod
        def resource_url_md5(cls):
            # Some SpikingJelly revisions map SSC to SHD resources by mistake.
            # Override resources here so SSC always downloads the correct files.
            return [
                ('ssc_train.h5.zip', 'https://zenkelab.org/datasets/ssc_train.h5.zip', 'd102be95e7144fcc0553d1f45ba94170'),
                ('ssc_valid.h5.zip', 'https://zenkelab.org/datasets/ssc_valid.h5.zip', 'b4eee3516a4a90dd0c71a6ac23a8ae43'),
                ('ssc_test.h5.zip', 'https://zenkelab.org/datasets/ssc_test.h5.zip', 'a35ff1e9cffdd02a20eb850c17c37748'),
            ]

        @classmethod
        def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
            # Use absolute targets to avoid relative-symlink breakage on some hosts.
            for f in Path(extract_root).iterdir():
                target = Path(raw_root) / f.name
                if target.exists():
                    continue
                target.symlink_to(f.resolve())

        def __getitem__(self, i: int):
            if self.data_type == 'event':
                events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
                label = self.h5_file['labels'][i]
                if self.transform is not None:
                    events = self.transform(events)
                if self.target_transform is not None:
                    label = self.target_transform(label)

                return events, label

            elif self.data_type == 'frame':
                local_fields_ok = hasattr(self, 'frames_path') and hasattr(self, 'frames_label')
                if local_fields_ok:
                    frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
                    label = self.frames_label[i]
                    apply_local_transform = True
                else:
                    # Newer SpikingJelly revisions changed internal frame bookkeeping.
                    frames, label = super().__getitem__(i)
                    apply_local_transform = False

                frames = _to_numpy_frames(frames, self.n_bins)
                binned_frames = _bin_frames(frames, self.n_bins)

                if apply_local_transform:
                    if self.transform is not None:
                        binned_frames = self.transform(binned_frames)
                    if self.target_transform is not None:
                        label = self.target_transform(label)

                return binned_frames, label
else:
    class BinnedSpikingSpeechCommands:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SpikingSpeechCommands is unavailable in installed spikingjelly version."
            )


def build_transform(is_train):
    if torchaudio is None:
        raise ImportError("torchaudio is required for GSC transforms.")
    sample_rate=16000
    window_size=256
    hop_length=80
    n_mels=140
    f_min=50
    f_max=14000

    t = [augmentations.PadOrTruncate(sample_rate),
         Resample(sample_rate, sample_rate // 2)]
    if is_train:
        t.extend([augmentations.RandomRoll(dims=(1,)),
                  augmentations.SpeedPerturbation(rates=(0.5, 1.5), p=0.5)
                 ])

    t.append(Spectrogram(n_fft=window_size, hop_length=hop_length, power=2))

    if is_train:
        pass

    t.extend([MelScale(n_mels=n_mels,
                       sample_rate=sample_rate // 2,
                       f_min=f_min,
                       f_max=f_max,
                       n_stft=window_size // 2 + 1),
              AmplitudeToDB()
             ])

    return transforms.Compose(t)

labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

target_transform = lambda word : torch.tensor(labels.index(word))

class GSpeechCommands(Dataset):
    def __init__(self, root, split_name, transform=None, target_transform=None, download=True):
        if SPEECHCOMMANDS is None:
            raise ImportError("torchaudio is required for GSpeechCommands.")

        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = SPEECHCOMMANDS(root, download=download, subset=split_name)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        waveform, _,label,_,_ = self.dataset.__getitem__(index)

        if self.transform is not None:
            waveform = self.transform(waveform).squeeze().t()

        target = label

        if self.target_transform is not None:
            target = self.target_transform(target)

        return waveform, target, torch.zeros(1)
