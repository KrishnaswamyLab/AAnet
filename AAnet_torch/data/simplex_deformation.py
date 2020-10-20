from torch.utils.data import Dataset

from ..utils import generate_data_on_sphere


class SimplexSphereProjection(Dataset):
    def __init__(self, n_obs=1000, radius=1):
        super().__init__()
        self.n_obs = n_obs
        self.radius = radius
        self.data, self.vertices = generate_data_on_sphere(self.n_obs, self.radius)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
