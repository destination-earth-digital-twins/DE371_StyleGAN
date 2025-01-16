import os
from datetime import datetime, timedelta
import numpy as np
import torch
from torch.utils.data import Dataset

def get_time_encoding(t, period):
    t_norm = t % period / period
    return (
        torch.sin(2 * torch.pi * t_norm),
        torch.cos(2 * torch.pi * t_norm)
    )

def get_day_of_year(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = date.timetuple().tm_yday
    return day_of_year

class InterpolatorDataset(Dataset):
    def __init__(self, start_date, end_date, latent_basepath, real_basepath,
                 leadtimes=np.arange(1, 46, 1), invstep=1000, dt=6, fmt='npy', include_input_leadtimes=False):
        self.start_date = start_date
        self.end_date = end_date
        self.latent_basepath = latent_basepath
        self.real_basepath = real_basepath
        self.leadtimes = leadtimes
        self.invstep = invstep
        self.dt = dt
        self.fmt = fmt
        self.include_input_leadtimes = include_input_leadtimes
        if len(self.leadtimes) >= 2:
            self.leadtime_step = self.leadtimes[1] - self.leadtimes[0]
        else:
            self.leadtime_step = None
        self.indices = self._build_index()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Return batch corresponding to the given index."""
        date_str, t_start, t_end, t_int = self.indices[idx]
        batch = self.get_batch(date_str, t_start, t_end, t_int)
        if batch is None:
            raise RuntimeError(f"Batch for index {idx} could not be loaded.")
        return batch

    def _build_index(self):
        """Builds the list of indices for the dataset."""
        indices = []
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        current_date = start_date

        t_int_min = 1
        t_int_max = self.dt

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            for t_start in range(self.leadtimes[0], self.leadtimes[-1] - self.dt + 1, self.leadtime_step):
                t_end = t_start + self.dt
                if self.include_input_leadtimes:
                    t_int_min = 0
                    t_int_max = self.dt + 1
                for t_int in range(t_int_min, t_int_max):
                    t_int = t_start + t_int
                    # Check if the required files exist
                    latent_files_exist = all(
                        self.file_exists(self.latent_basepath, date_str, leadtime, invstep=self.invstep)
                        for leadtime in [t_start, t_int, t_end]
                    )
                    real_files_exist = all(
                        self.file_exists(self.real_basepath, date_str, leadtime)
                        for leadtime in [t_start, t_int, t_end]
                    )
                    if latent_files_exist and real_files_exist:
                        indices.append([date_str, t_start, t_end, t_int])
            current_date += timedelta(days=1)

        return indices

    def file_exists(self, basepath, date, leadtime, invstep=None):
        """Check if the required file exists."""
        if invstep:
            filename = f"{basepath}_{date}_{leadtime}_{invstep}.{self.fmt}"
        else:
            filename = f"{basepath}_{date}_{leadtime}.{self.fmt}"
        return os.path.exists(filename)

    def get_sample(self, basepath, date, leadtime, invstep=None):
        """Loads a single sample (returns None if loading fails)."""
        if invstep:
            filename = f"{basepath}_{date}_{leadtime}_{invstep}.{self.fmt}"
        else:
            filename = f"{basepath}_{date}_{leadtime}.{self.fmt}"

        if os.path.exists(filename):
            try:
                sample = np.load(filename)
                return torch.from_numpy(sample)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        return None

    def get_batch(self, date, start_leadtime, end_leadtime, int_leadtime):
        """Returns a batch of tensors (None if any part is missing)."""
        try:
            w_start = self.get_sample(self.latent_basepath, date, start_leadtime, self.invstep)
            w_t = self.get_sample(self.latent_basepath, date, int_leadtime, self.invstep)
            w_end = self.get_sample(self.latent_basepath, date, end_leadtime, self.invstep)
            r_start = self.get_sample(self.real_basepath, date, start_leadtime)
            r_t = self.get_sample(self.real_basepath, date, int_leadtime)
            r_end = self.get_sample(self.real_basepath, date, end_leadtime)

            if None in [w_start, w_t, w_end, r_start, r_t, r_end]:
                print(f"Missing data for {date} ({start_leadtime}, {end_leadtime}, {int_leadtime})")
                return None

            assert len(w_start) == len(w_t) == len(w_end) == len(r_start) == len(r_t) == len(r_end)

            t_frac = torch.tensor((int_leadtime - start_leadtime) / self.dt, dtype=torch.float32)
            t_start_encoding = get_time_encoding(torch.tensor(start_leadtime), 24)
            t_end_encoding = get_time_encoding(torch.tensor(end_leadtime), 24)
            t_int_encoding = get_time_encoding(torch.tensor(int_leadtime), 24)
            day_encoding = get_time_encoding(torch.tensor(get_day_of_year(date)), 366)
            t_encodings = torch.tensor([
                *t_start_encoding, *t_end_encoding, *t_int_encoding, *day_encoding
            ])
            assert t_encodings.max() <= 1
            assert t_encodings.min() >= -1
            
            return w_start, w_end, t_frac, t_encodings, w_t, r_start, r_end, r_t

        except Exception as e:
            print(f"Error creating batch for {date} ({start_leadtime}, {end_leadtime}, {int_leadtime}): {e}")
            return None