import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()
torch.set_default_dtype(torch.float32)

@typechecked
class Sequence:
    def __init__(
        self,
        time: np.ndarray | TensorType[float, "events"],
        checkins: np.ndarray | TensorType[int, "events"],
        category: np.ndarray | TensorType[int, "events"],
        condition1: np.ndarray | TensorType[int, "events"],
        condition2: np.ndarray | TensorType[int, "events"],
        condition3: np.ndarray | TensorType[int, "events"],
        condition4: np.ndarray | TensorType[int, "events"],
        condition5: np.ndarray | TensorType[int, "events"],
        condition6: np.ndarray | TensorType[int, "events"],
        condition1_indicator: np.ndarray | TensorType[int, "granularity"],
        condition2_indicator: np.ndarray | TensorType[int, "granularity"],
        condition3_indicator: np.ndarray | TensorType[int, "granularity"],
        condition4_indicator: np.ndarray | TensorType[int, "granularity"],
        condition5_indicator: np.ndarray | TensorType[int, "granularity"],
        condition6_indicator: np.ndarray | TensorType[int, "granularity"],
        tmax: Union[np.ndarray, TensorType[float], float],
        device: Union[torch.device, str] = "cpu",
        kept_points: Union[np.ndarray, TensorType, None] = None,

    ) -> None:
        super().__init__()
        if not isinstance(time, torch.Tensor):
            time = torch.as_tensor(time)

        if not isinstance(checkins, torch.Tensor):
            checkins = torch.as_tensor(checkins)

        if not isinstance(category, torch.Tensor):
            category = torch.as_tensor(category)

        if not isinstance(condition1, torch.Tensor):
            condition1 = torch.as_tensor(condition1)

        if not isinstance(condition2, torch.Tensor):
            condition2 = torch.as_tensor(condition2)

        if not isinstance(condition3, torch.Tensor):
            condition3 = torch.as_tensor(condition3)

        if not isinstance(condition4, torch.Tensor):
            condition4 = torch.as_tensor(condition4)

        if not isinstance(condition5, torch.Tensor):
            condition5 = torch.as_tensor(condition5)
            
        if not isinstance(condition6, torch.Tensor):
            condition6 = torch.as_tensor(condition6)

        if not isinstance(condition1_indicator, torch.Tensor):
            condition1_indicator = torch.as_tensor(condition1_indicator)

        if not isinstance(condition2_indicator, torch.Tensor):
            condition2_indicator = torch.as_tensor(condition2_indicator)
            
        if not isinstance(condition3_indicator, torch.Tensor):
            condition3_indicator = torch.as_tensor(condition3_indicator)

        if not isinstance(condition4_indicator, torch.Tensor):
            condition4_indicator = torch.as_tensor(condition4_indicator)

        if not isinstance(condition5_indicator, torch.Tensor):
            condition5_indicator = torch.as_tensor(condition5_indicator)

        if not isinstance(condition6_indicator, torch.Tensor):
            condition6_indicator = torch.as_tensor(condition6_indicator)

        if tmax is not None:
            if not isinstance(tmax, torch.Tensor):
                tmax = torch.as_tensor(tmax)

        if kept_points is not None:
            if not isinstance(kept_points, torch.Tensor):
                kept_points = torch.as_tensor(kept_points)
            kept_points = kept_points

        self.time = time
        self.checkins = checkins
        self.category = category

        self.condition1 = condition1
        self.condition2 = condition2
        self.condition3 = condition3
        self.condition4 = condition4
        self.condition5 = condition5
        self.condition6 = condition6

        self.condition1_indicator = condition1_indicator
        self.condition2_indicator = condition2_indicator
        self.condition3_indicator = condition3_indicator
        self.condition4_indicator = condition4_indicator
        self.condition5_indicator = condition5_indicator
        self.condition6_indicator = condition6_indicator        

        self.tmax = tmax
        self.kept_points = kept_points

        self.device = device
        self.to(device)
        tau = torch.diff(
            self.time,
            prepend=torch.as_tensor([0.0], device=device),
            append=torch.as_tensor([self.tmax], device=device),
        )

        self.tau = tau

        self.tmax

    def __len__(self) -> int:
        return len(self.time)

    def __getitem__(self, key: str):
        return getattr(self, key, None)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def keys(self) -> List[str]:
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def to(self, device: Union[str, torch.device]) -> "Sequence":
        self.device = device
        for key in self.keys():
            if key != "device":
                self[key] = self[key].to(device)
        return self


@typechecked
class Batch:
    def __init__(
        self,
        mask: TensorType[bool, "batch", "sequence"],
        time: TensorType[float, "batch", "sequence"],
        condition1: TensorType[int, "batch", "sequence"],
        condition2: TensorType[int, "batch", "sequence"],
        condition3: TensorType[int, "batch", "sequence"],
        condition4: TensorType[int, "batch", "sequence"],
        condition5: TensorType[int, "batch", "sequence"],
        condition6: TensorType[int, "batch", "sequence"],
        condition1_indicator: TensorType[int, "batch", "granularity"],
        condition2_indicator: TensorType[int, "batch", "granularity"],
        condition3_indicator: TensorType[int, "batch", "granularity"],
        condition4_indicator: TensorType[int, "batch", "granularity"],
        condition5_indicator: TensorType[int, "batch", "granularity"],
        condition6_indicator: TensorType[int, "batch", "granularity"],
        tau: TensorType[float, "batch", "sequence"],
        tmax: TensorType[float],
        unpadded_length: TensorType[int, "batch"],
        kept: Union[TensorType, None] = None,
        checkin_sequences: Union[TensorType, None] = None,
        category_mask: Union[TensorType, None] = None,
        poi_mask: Union[TensorType, None] = None,
    ):
        super().__init__()
        self.time = time
        self.condition1 = condition1
        self.condition2 = condition2
        self.condition3 = condition3
        self.condition4 = condition4
        self.condition5 = condition5
        self.condition6 = condition6

        self.condition1_indicator = condition1_indicator
        self.condition2_indicator = condition2_indicator
        self.condition3_indicator = condition3_indicator
        self.condition4_indicator = condition4_indicator
        self.condition5_indicator = condition5_indicator
        self.condition6_indicator = condition6_indicator  

        self.tau = tau
        self.tmax = tmax
        self.kept = kept

        # Padding and mask
        self.unpadded_length = unpadded_length
        self.mask = mask

        self.checkin_sequences = checkin_sequences
        self.category_mask = category_mask
        self.poi_mask = poi_mask
        self._validate()

    @property
    def batch_size(self) -> int:
        return self.time.shape[0]

    @property
    def seq_len(self) -> int:
        return self.time.shape[1]

    @property
    def content_len(self) -> int:
        if self.checkin_sequences is not None:
            assert self.checkin_sequences.shape[1] == max(self.unpadded_length)*2+3, "wrong content_len"
            return self.checkin_sequences.shape[1]
        return (max(self.unpadded_length)*2+3).item() if isinstance(max(self.unpadded_length)*2+3, torch.Tensor) else max(self.unpadded_length)*2+3
        
    def __len__(self):
        return self.batch_size

    def __getitem__(self, key: str):
        return getattr(self, key, None)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def keys(self) -> List[str]:
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def to(self, device: Union[str, torch.device]) -> "Batch":
        self.device = device
        for key in self.keys():
            if key != "device":
                self[key] = self[key].to(device)
        return self

    @staticmethod
    def from_sequence_list(sequences: List[Sequence]) -> "Batch":
        """
        Create batch from list of sequences.
        """
        # Pad sequences for batching
        tmax = torch.cat(
            [sequence.tmax.unsqueeze(dim=0) for sequence in sequences], dim=0
        ).max()
        tau = pad([sequence.tau for sequence in sequences])
        time = pad(
            [sequence.time for sequence in sequences], length=tau.shape[-1]
        )

        condition1 = pad(
            [sequence.condition1 for sequence in sequences], length=tau.shape[-1]
        )

        condition2 = pad(
            [sequence.condition2 for sequence in sequences], length=tau.shape[-1]
        )

        condition3 = pad(
            [sequence.condition3 for sequence in sequences], length=tau.shape[-1]
        )

        condition4 = pad(
            [sequence.condition4 for sequence in sequences], length=tau.shape[-1]
        )

        condition5 = pad(
            [sequence.condition5 for sequence in sequences], length=tau.shape[-1]
        )

        condition6 = pad(
            [sequence.condition6 for sequence in sequences], length=tau.shape[-1]
        )

        condition1_indicator = torch.stack([sequence.condition1_indicator for sequence in sequences])

        condition2_indicator = torch.stack([sequence.condition2_indicator for sequence in sequences])

        condition3_indicator = torch.stack([sequence.condition3_indicator for sequence in sequences])

        condition4_indicator = torch.stack([sequence.condition4_indicator for sequence in sequences])

        condition5_indicator = torch.stack([sequence.condition5_indicator for sequence in sequences])

        condition6_indicator = torch.stack([sequence.condition6_indicator for sequence in sequences])


        device = tau.device

        sequence_length = torch.tensor(
            [len(sequence) for sequence in sequences], device=device
        )

        if sequences[0].kept_points != None:
            kept_points = pad(
                [sequence.kept_points for sequence in sequences],
                length=tau.shape[-1],
            )
        else:
            kept_points = None

        # Compute event mask for batching
        mask = (
            torch.arange(0, tau.shape[-1], device=device)[None, :]
            < sequence_length[:, None]
        )

        #Get data for discrete diffsion start=0, |=1, end=2, pad=3
        if sequences[0].category is not None:
            checkin_sequences = pad(
                [torch.cat((torch.tensor([0],device=device), sequences[idx].category, torch.tensor([1],device=device), sequences[idx].checkins, torch.tensor([2],device=device)), dim=0)for idx in range(len(sequences))]
            ,value=3)

        category_mask= pad(
            [torch.cat((torch.tensor([0],device=device,dtype=torch.int64), torch.tensor([1]* len(sequences[idx].time),device=device,dtype=torch.int64), torch.tensor([0]* (len(sequences[idx].time)+2),device=device,dtype=torch.int64)), dim=0)for idx in range(len(sequences))]
        )
        poi_mask= pad(
            [torch.cat((torch.tensor([0]* (len(sequences[idx].time)+2),device=device,dtype=torch.int64), torch.tensor([1]*len(sequences[idx].time),device=device,dtype=torch.int64),torch.tensor([0],device=device,dtype=torch.int64)), dim=0)for idx in range(len(sequences))]
        )

        batch = Batch(
            mask=mask,
            time=time,
            checkin_sequences=checkin_sequences,
            category_mask=category_mask,
            poi_mask=poi_mask,
            condition1=condition1,
            condition2=condition2,
            condition3=condition3,
            condition4=condition4,
            condition5=condition5,
            condition6=condition6,
            condition1_indicator=condition1_indicator,
            condition2_indicator=condition2_indicator,
            condition3_indicator=condition3_indicator,
            condition4_indicator=condition4_indicator,
            condition5_indicator=condition5_indicator,
            condition6_indicator=condition6_indicator,
            tau=tau,
            tmax=tmax,
            unpadded_length=sequence_length,
            kept=kept_points,
        )
        return batch

    def add_events(self, other: "Batch") -> "Batch":
        """
        Add batch of events to sequences.

        Parameters:
        ----------
        other : Batch
            Batch of events to add.

        Returns:
        -------
        Batch
            Batch of events with added events.
        """
        assert len(other) == len(
            self
        ), "The number of sequences to add does not match the number of sequences in the batch."
        other = other.to(self.time.device)
        tmax = max(self.tmax, other.tmax)

        if self.kept is None:
            kept = torch.cat(
                [
                    torch.ones_like(self.time, dtype=bool),
                    torch.zeros_like(other.time, dtype=bool),
                ],
                dim=-1,
            )
        else:
            kept = torch.cat(
                [self.kept, torch.zeros_like(other.time, dtype=bool)],
                dim=-1,
            )

        return self.remove_unnescessary_padding(
            time=torch.cat([self.time, other.time], dim=-1),
            condition1=torch.cat([self.condition1, other.condition1], dim=-1),
            condition2=torch.cat([self.condition2, other.condition2], dim=-1),
            condition3=torch.cat([self.condition3, other.condition3], dim=-1),
            condition4=torch.cat([self.condition4, other.condition4], dim=-1),
            condition5=torch.cat([self.condition5, other.condition5], dim=-1),
            condition6=torch.cat([self.condition6, other.condition6], dim=-1),
            condition1_indicator=self.condition1_indicator,
            condition2_indicator=self.condition2_indicator,
            condition3_indicator=self.condition3_indicator,
            condition4_indicator=self.condition4_indicator,
            condition5_indicator=self.condition5_indicator,
            condition6_indicator=self.condition6_indicator,
            mask=torch.cat([self.mask, other.mask], dim=-1),
            kept=kept,
            tmax=tmax,
        )

    def to_time_list(self):
        time = []
        for i in range(len(self)):
            time.append(self.time[i][self.mask[i]].detach().cpu().numpy())
        return time


    def to_seq_list(self,gps_dict):
        seqs_new = []
        for i in range(len(self)):
            gps=[]
            index=[]
            arrival_times_gen=self.time[i][self.mask[i]].detach().cpu().numpy()
            marks_gen=self.checkin_sequences[i][self.category_mask[i].bool()].detach().cpu().numpy()
            condition1_gen=self.condition1[i][self.mask[i]].detach().cpu().numpy()
            condition2_gen=self.condition2[i][self.mask[i]].detach().cpu().numpy()
            condition3_gen=self.condition3[i][self.mask[i]].detach().cpu().numpy()
            condition4_gen=self.condition4[i][self.mask[i]].detach().cpu().numpy()
            condition5_gen=self.condition5[i][self.mask[i]].detach().cpu().numpy()
            condition6_gen=self.condition6[i][self.mask[i]].detach().cpu().numpy()
            checkins_gen = self.checkin_sequences[i][self.poi_mask[i].bool()].detach().cpu().numpy()

            for idx in range(len(checkins_gen)):
                if checkins_gen[idx] not in gps_dict.keys():
                    index.append(idx)
                    continue
                gps_str=gps_dict[checkins_gen[idx]].split(',')
                gps.append([float(gps_str[0]),float(gps_str[1])])

            arrival_times_clean = np.delete(arrival_times_gen, index)
            marks_clean = np.delete(marks_gen, index)
            checkins_clean = np.delete(checkins_gen, index)
            condition1_clean = np.delete(condition1_gen, index)
            condition2_clean = np.delete(condition2_gen, index)
            condition3_clean= np.delete(condition3_gen, index)
            condition4_clean= np.delete(condition4_gen, index)
            condition5_clean = np.delete(condition5_gen, index)
            condition6_clean = np.delete(condition6_gen, index)

            seqs_new.append({'arrival_times': arrival_times_clean,
                             'marks': marks_clean,
                             'checkins': checkins_clean,
                             'gps':gps,
                             'condition1':condition1_clean,
                             'condition2':condition2_clean,
                             'condition3':condition3_clean,
                             'condition4':condition4_clean,
                             'condition5':condition5_clean,
                             'condition6':condition6_clean,
                             'condition1_indicator':self.condition1_indicator[i].detach().cpu().numpy(),
                             'condition2_indicator':self.condition2_indicator[i].detach().cpu().numpy(),
                             'condition3_indicator':self.condition3_indicator[i].detach().cpu().numpy(),
                             'condition4_indicator':self.condition4_indicator[i].detach().cpu().numpy(),
                             'condition5_indicator':self.condition5_indicator[i].detach().cpu().numpy(),
                             'condition6_indicator':self.condition6_indicator[i].detach().cpu().numpy()})      
        return seqs_new

    def mask_check(self):
        if self.checkin_sequences is None:
            self.category_mask= pad(
                [torch.cat((torch.tensor([0],device=self.time.device,dtype=torch.int64), torch.tensor([1]* length ,device=self.time.device,dtype=torch.int64), torch.tensor([0]* (length+2),device=self.time.device,dtype=torch.int64)), dim=0) for length in self.unpadded_length]
            )
            self.poi_mask= pad(
                [torch.cat((torch.tensor([0]* (length+2),device=self.time.device,dtype=torch.int64), torch.tensor([1]*length, device=self.time.device,dtype=torch.int64),torch.tensor([0],device=self.time.device,dtype=torch.int64)), dim=0) for length in self.unpadded_length]
            )
            # self.checkin_sequences = self.poi_mask | self.category_mask
        self.device = self.time.device
        return self
    
    @staticmethod
    def sort_time(
        time: TensorType[float, "batch", "sequence"], 
        condition1: TensorType[int, "batch", "sequence"], 
        condition2: TensorType[int, "batch", "sequence"], 
        condition3: TensorType[int, "batch", "sequence"], 
        condition4: TensorType[int, "batch", "sequence"], 
        condition5: TensorType[int, "batch", "sequence"], 
        condition6: TensorType[int, "batch", "sequence"], 
        mask: TensorType[bool, "batch", "sequence"], 
        kept, 
        tmax: TensorType[float]
    ):
        """
        Sort events by time.

        Returns:
        -------
        time : TensorType[float, "batch", "sequence"]
            Tensor of event times.
        mask : TensorType[bool, "batch", "sequence"]
            Tensor of event masks.
        kept : TensorType[bool, "batch", "sequence"]
            Tensor indicating kept events.
        """
        # Sort time and mask by time
        time[~mask] = 2 * tmax
        sort_idx = torch.argsort(time, dim=-1)
        mask = torch.take_along_dim(mask, sort_idx, dim=-1)
        time = torch.take_along_dim(time, sort_idx, dim=-1)
        condition1 = torch.take_along_dim(condition1, sort_idx, dim=-1)
        condition2 = torch.take_along_dim(condition2, sort_idx, dim=-1)
        condition3 = torch.take_along_dim(condition3, sort_idx, dim=-1)
        condition4 = torch.take_along_dim(condition4, sort_idx, dim=-1)
        condition5 = torch.take_along_dim(condition5, sort_idx, dim=-1)
        condition6 = torch.take_along_dim(condition6, sort_idx, dim=-1)
        if kept is not None:
            kept = torch.take_along_dim(kept, sort_idx, dim=-1)
        else:
            kept = None
        time = time * mask

        condition1 = condition1 * mask
        condition2 = condition2 * mask
        condition3 = condition3 * mask
        condition4 = condition4 * mask
        condition5 = condition5 * mask
        condition6 = condition6 * mask

        return time, condition1, condition2, condition3, condition4, condition5, condition6, mask, kept

    @staticmethod
    def remove_unnescessary_padding(
        time: TensorType[float, "batch", "sequence"],
        condition1: TensorType[int, "batch", "sequence"],
        condition2: TensorType[int, "batch", "sequence"],
        condition3: TensorType[int, "batch", "sequence"],
        condition4: TensorType[int, "batch", "sequence"],
        condition5: TensorType[int, "batch", "sequence"],
        condition6: TensorType[int, "batch", "sequence"],
        condition1_indicator: TensorType[int, "batch", "granularity"],
        condition2_indicator:  TensorType[int, "batch", "granularity"],
        condition3_indicator:  TensorType[int, "batch", "granularity"],
        condition4_indicator:  TensorType[int, "batch", "granularity"],
        condition5_indicator:  TensorType[int, "batch", "granularity"],
        condition6_indicator:  TensorType[int, "batch", "granularity"], 
        mask: TensorType[bool, "batch", "sequence"], 
        kept, 
        tmax
    ):
        """
        Remove unnescessary padding from batch.

        Returns:
        -------
        Batch
            Batch of events without unnescessary padding.
        """
        # Sort by time
        time, condition1, condition2, condition3, condition4, condition5, condition6, mask, kept = Batch.sort_time(time, condition1, condition2, condition3, condition4, condition5, condition6, mask, kept, tmax=tmax)

        # Reduce padding along sequence length
        max_length = max(mask.sum(-1)).int()
        mask = mask[:, : max_length + 1]
        time = time[:, : max_length + 1]
        condition1 = condition1[:, : max_length + 1]
        condition2 = condition2[:, : max_length + 1]
        condition3 = condition3[:, : max_length + 1]
        condition4 = condition4[:, : max_length + 1]
        condition5 = condition5[:, : max_length + 1]
        condition6 = condition6[:, : max_length + 1]
        if kept is not None:
            kept = kept[:, : max_length + 1]

        # compute interevent times
        time_tau = torch.where(mask, time, tmax)
        tau = torch.diff(
            time_tau, prepend=torch.zeros_like(time_tau)[:, :1], dim=-1
        )
        tau = tau * mask

        return Batch(
            mask=mask,
            time=time,
            condition1=condition1,
            condition2=condition2,
            condition3=condition3,
            condition4=condition4,
            condition5=condition5,
            condition6=condition6,
            condition1_indicator=condition1_indicator,
            condition2_indicator=condition2_indicator,
            condition3_indicator=condition3_indicator,
            condition4_indicator=condition4_indicator,
            condition5_indicator=condition5_indicator,
            condition6_indicator=condition6_indicator,
            tau=tau,
            tmax=tmax,
            unpadded_length=mask.sum(-1).long(),
            kept=kept,
        )

    def thin(self, alpha: TensorType[float]) -> Tuple["Batch", "Batch"]:
        """
        Thin events according to alpha.

        Parameters:
        ----------
        alpha : TensorType[float]
            Probability of keeping an event.

        Returns:
        -------
        keep : Batch
            Batch of kept events.
        remove : Batch
            Batch of removed events.
        """
        if alpha.dim() == 1:
            keep = torch.bernoulli(
                alpha.unsqueeze(1).repeat(1, self.seq_len)
            ).bool()
        elif alpha.dim() == 2:
            keep = torch.bernoulli(alpha).bool()
        else:
            raise Warning("alpha has too many dimensions")

        # remove from mask
        keep_mask = self.mask * keep
        rem_mask = self.mask * ~keep

        # shorten padding after removal
        return self.remove_unnescessary_padding(
            time=self.time * keep_mask,
            condition1 = self.condition1 * keep_mask,
            condition2 = self.condition2 * keep_mask,
            condition3 = self.condition3 * keep_mask,
            condition4 = self.condition4 * keep_mask,
            condition5 = self.condition5 * keep_mask,
            condition6 = self.condition6 * keep_mask,
            condition1_indicator=self.condition1_indicator,
            condition2_indicator=self.condition2_indicator,
            condition3_indicator=self.condition3_indicator,
            condition4_indicator=self.condition4_indicator,
            condition5_indicator=self.condition5_indicator,
            condition6_indicator=self.condition6_indicator,
            mask=keep_mask,
            kept=self.kept * keep_mask if self.kept is not None else self.kept,
            tmax=self.tmax,
        ), self.remove_unnescessary_padding(
            time=self.time * rem_mask,
            condition1 = self.condition1 * rem_mask,
            condition2 = self.condition2 * rem_mask,
            condition3 = self.condition3 * rem_mask,
            condition4 = self.condition4 * rem_mask,
            condition5 = self.condition5 * rem_mask,
            condition6 = self.condition6 * rem_mask,
            condition1_indicator=self.condition1_indicator,
            condition2_indicator=self.condition2_indicator,
            condition3_indicator=self.condition3_indicator,
            condition4_indicator=self.condition4_indicator,
            condition5_indicator=self.condition5_indicator,
            condition6_indicator=self.condition6_indicator,
            mask=rem_mask,
            kept=self.kept * rem_mask if self.kept is not None else self.kept,
            tmax=self.tmax,
        )


    def _validate(self):
        """
        Validate batch, esp. masking.
        """
        # Check mask
        # mask as long as seq len;
        assert (self.mask.sum(-1) == self.unpadded_length).all(), "wrong mask"
        assert (self.time * self.mask == self.time).all(), "wrong mask"

        assert torch.allclose(
            self.tau.cumsum(-1) * self.mask, self.time * self.mask, atol=1e-5
        ), "wrong tau"

        assert self.tau.shape == (
            self.batch_size,
            self.seq_len,
        ), f"tau has wrong shape {self.tau.shape}, expected {(self.batch_size, self.seq_len)}"

        if self.checkin_sequences != None:
            assert (self.category_mask.sum(-1) == self.unpadded_length).all(), "wrong mask"
            assert (self.poi_mask.sum(-1) == self.unpadded_length).all(), "wrong mask"
            assert self.checkin_sequences.shape == self.category_mask.shape ==self.poi_mask.shape, "wrong mask"

@typechecked
def pad(sequences, length: Union[int, None] = None, value: float = 0):
    """
    Utility function to generate padding and mask for sequences.
    Parameters:
    ----------
            sequences: List of sequences.
            value: float = 0,
            length: Optional[int] = None,
    Returns:
    ----------
            sequences: Padded sequence,
                shape (batch_size, seq_length)
            mask: Boolean mask that indicates which entries
                do NOT correspond to padding, shape (batch_size, seq_len)
    """

    # Pad first sequence to enforce padding length
    if length:
        device = sequences[0].device
        dtype = sequences[0].dtype
        tensor_length = sequences[0].size(0)
        intial_pad = torch.empty(
            torch.Size([length]) + sequences[0].shape[1:],
            dtype=dtype,
            device=device,
        ).fill_(value)
        intial_pad[:tensor_length, ...] = sequences[0]
        sequences[0] = intial_pad

    sequences = pad_sequence(
        sequences, batch_first=True, padding_value=value
    )  # [order]

    return sequences


@typechecked
class SequenceDataset(torch.utils.data.Dataset):
    """Dataset of variable-length event sequences."""

    def __init__(
        self,
        sequences: List[Sequence],
    ):
        self.sequences = sequences
        self.tmax = sequences[0].tmax

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        return sequence

    def __len__(self) -> int:
        return len(self.sequences)

    def to(self, device: Union[torch.device, str]):
        for sequence in self.sequences:
            sequence.to(device)


@typechecked
class DataModule(pl.LightningDataModule):
    """
    Datamodule for variable length event sequences for temporal point processes.

    Parameters:
    ----------
    root : str
        Path to data.
    name : str
        Name of dataset.
    split_seed : int
        Seed for random split.
    batch_size : int
        Batch size.
    train_size : float
        Percentage of data to use for training.
    forecast : bool
        Whether to use the dataset for forecasting.
    """

    def __init__(
        self,
        root: str,
        name: str,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.name = name

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self) -> None:
        """Load sequence data from root."""
        time_sequences_train, num_category, num_poi, gps_dict  = load_sequences(Path(self.root + f'/{self.name}'), self.name + '_train')
        time_sequences_test, _, _, _ = load_sequences(Path(self.root + f'/{self.name}'), self.name + '_test')

        self.train_data = SequenceDataset(sequences=time_sequences_train)
        self.test_data = SequenceDataset(sequences=time_sequences_test)
        self.tmax = self.train_data.tmax
        self.num_category = num_category
        self.num_poi = num_poi
        self.gps_dict = gps_dict

        self.get_statistics()

    def get_statistics(self):
        # Get train stats
        seq_lengths = []
        for i in range(len(self.train_data)):
            seq_lengths.append(len(self.train_data[i]))
        self.n_max = max(seq_lengths)

    def setup(self, stage=None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=Batch.from_sequence_list,
            num_workers=0,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=Batch.from_sequence_list,
            num_workers=0,
            drop_last=False,
        )


def load_sequences(root, name: str) -> List[Sequence]:
    """Load dataset.

    Parameters:
    ----------
    root : str
        Path to data.
    name : str
        Name of dataset.

    Returns:
    -------
    time_sequences : List[Sequence]
        List of event sequences.

    """
    path = os.path.join(root, f"{name}.pkl")
    loader = torch.load(path, map_location=torch.device("cpu"))

    sequences = loader["sequences"]
    tmax = loader["t_max"]
    num_category = loader["num_marks"]
    num_poi = loader["num_pois"]
    gps_dict = loader["poi_gps"]

    time_sequences = [
        Sequence(
            time = seq["arrival_times"],
            condition1 = seq["condition1"],
            condition2 = seq["condition2"],
            condition3 = seq["condition3"],
            condition4 = seq["condition4"],
            condition5 = seq["condition5"],
            condition6 = seq["condition6"],
            condition1_indicator = seq["condition1_indicator"],
            condition2_indicator = seq["condition2_indicator"],
            condition3_indicator = seq["condition3_indicator"],
            condition4_indicator = seq["condition4_indicator"],
            condition5_indicator = seq["condition5_indicator"],
            condition6_indicator = seq["condition6_indicator"],
            tmax=tmax,
            checkins = seq["checkins"],
            category = seq["marks"],
        )
        for seq in sequences
    ]
    return time_sequences, num_category, num_poi, gps_dict
