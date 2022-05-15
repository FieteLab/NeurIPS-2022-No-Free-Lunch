import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.spatial.distance
import seaborn as sns
import torch
import torch.nn
import torch.nn.parameter
import torch.optim

# Set seeds.
np.random.seed(0)
torch.manual_seed(0)

exp_dir = 'notebooks_v2/30_banino_exploration'
num_grid_cells = 4096
num_place_cells = 512
num_spatial_positions = 1000
num_grad_steps = 10000
sigma = 0.12
sigma_squared = np.square(sigma)

place_cell_locations = np.random.uniform(
    low=-1.1,
    high=1.1,
    size=(num_place_cells, 2))

spatial_locations = np.random.uniform(
    low=-1.1,
    high=1.1,
    size=(num_spatial_positions, 2))

# Shape: (num spatial points, num place cells)
distances = scipy.spatial.distance.cdist(
    XA=spatial_locations,
    XB=place_cell_locations)

# Shape: (num place cells, num spatial points)
pc_activity_numpy = np.exp(-0.5 * distances / sigma_squared)
pc_activity_numpy /= np.sum(pc_activity_numpy, axis=0, keepdims=True)

# Shape: (num spatial points, num spatial points)
pc_second_moment_matrix_numpy = np.matmul(pc_activity_numpy, pc_activity_numpy.T)


# plt.imshow(place_cell_second_moment_matrix)
# plt.show()

# sns.heatmap(place_cell_second_moment_matrix)
# plt.show()


class BaninoGridCells(torch.nn.Module):
    def __init__(self,
                 use_dropout_layer: bool = True):
        super().__init__()
        self.g = torch.nn.parameter.Parameter(
            data=torch.zeros(num_spatial_positions, num_grid_cells))
        self.use_dropout_layer = use_dropout_layer
        self.linear1 = torch.nn.Linear(
            in_features=num_grid_cells,
            out_features=num_place_cells)
        if self.use_dropout_layer:
            self.linear2 = torch.nn.Linear(
                in_features=num_place_cells,
                out_features=num_place_cells)
            self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, ):
        h1 = self.linear1(self.g)
        if self.use_dropout_layer:
            h1_dropout = self.dropout(h1)
            output = self.linear2(h1_dropout)
        else:
            output = h1
        return output


pc_activity_torch = torch.from_numpy(pc_activity_numpy)
pc_second_moment_matrix_torch = torch.from_numpy(pc_second_moment_matrix_numpy)
banino_gcs = BaninoGridCells(use_dropout_layer=True)
optimizer = torch.optim.SGD(banino_gcs.parameters(), lr=1e-5)
loss_per_grad_step = np.zeros(num_grad_steps)
for grad_step in range(num_grad_steps):
    optimizer.zero_grad()
    predicted_pc_activity_torch = banino_gcs.forward()
    loss = torch.sum(torch.square(predicted_pc_activity_torch - pc_activity_torch))
    print(f'Grad step: {grad_step}\t Loss: {loss.item()}')
    loss.backward()
    optimizer.step()
    loss_per_grad_step[grad_step] = loss.item()
    # Ensure each grid cell has unit norm.
    # banino_gcs.g.data has shape (num spatial locations, num grid cells)
    banino_gcs.g.data /= torch.sum(banino_gcs.g.data, dim=0, keepdim=True)

    if grad_step % 1000 == 0:
        joblib.dump(
            {'num_grid_cells': num_grid_cells,
             'num_place_cells': num_place_cells,
             'num_spatial_positions': num_spatial_positions,
             'sigma': sigma,
             'place_cell_locations': place_cell_locations,
             'spatial_locations': spatial_locations,
             'g': banino_gcs.g.data.numpy(),
             'W_1': banino_gcs.linear1.weight.data.numpy(),
             'b_1': banino_gcs.linear1.bias.data.numpy(),
             'W_2': banino_gcs.linear2.weight.data.numpy(),
             'b_2': banino_gcs.linear2.bias.data.numpy(),
             'loss_per_grad_step': loss_per_grad_step,
             'grad_step': grad_step
             },
            filename=os.path.join(exp_dir, f'ckpt_grad_step={grad_step}.joblib'))

print(10)
plt.close()
plt.plot(1 + np.arange(len(loss_per_grad_step)),
         loss_per_grad_step)
plt.yscale('log')
plt.ylabel(r'$|| A_2(d(A_1(G))) - P ||_F^2$')
plt.xlabel('Grad Step')
plt.show()
print(11)
