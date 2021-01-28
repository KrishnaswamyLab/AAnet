import numpy as np
import torch

def _barycentric_to_euclidean(X_barycentric, simplex_radius):
    '''
    Converts an M x (N + 1) array of barycentric coordinates into an M x N array
    of the corresponding Euclidean coordinates.
    '''

    n_dim = X_barycentric.shape[1] - 1
    coord_transform = np.zeros((n_dim, n_dim + 1))

    # Construct the N x (N + 1) coordinate transformation matrix
    # See the 'Simplex' Wiki page for a detailed explanation
    np.fill_diagonal(coord_transform, 1 / np.sqrt(2))
    for i in range(n_dim):
        for j in range(n_dim):
            coord_transform[i, j] -= 1 / (n_dim * np.sqrt(2)) * (1 + 1 / np.sqrt(n_dim + 1))
    for i in range(n_dim):
        coord_transform[i, n_dim] = 1 / (2 * np.sqrt(n_dim + 1))

    # Transform the simplex coordinates into points in R^n
    X_euclidean = simplex_radius * coord_transform @ X_barycentric.T
    X_euclidean = X_euclidean.T

    return X_euclidean

def _sample_from_n_simplex(n_dim, simplex_radius, n_samples):
    '''
    Samples Eucliean points uniformly from an N-dimensional simplex of a
    specified radius.
    '''

    exp_samples = np.random.exponential(size=(n_dim, n_samples))
    X_barycentric = exp_samples / np.linalg.norm(exp_samples, ord=1, axis=0)
    X_barycentric = X_barycentric.T

    X_euclidean = _barycentric_to_euclidean(X_barycentric, simplex_radius)

    return X_euclidean

def _stereographic_inverse(X_hyperplane):
    '''
    Computes the inverse of a stereographic projection, mapping points in N-dimensional
    Euclidean space to points in (N + 1)-dimensional Euclidean space on the unit hypersphere.
    '''

    X_hyperplane = X_hyperplane.T
    X_sphere = np.zeros((X_hyperplane.shape[0] + 1, X_hyperplane.shape[1]))

    norms = np.linalg.norm(X_hyperplane, axis=0)
    X_sphere[0] = (norms**2 - 1) / (norms**2 + 1)
    for i in range(1, X_sphere.shape[0]):
        X_sphere[i] = 2 * X_hyperplane[i - 1] / (norms**2 + 1)

    return X_sphere.T

def sample_from_stereo_sphere_simplex(n_dim=3, simplex_radius=1.0, n_samples=1000):
    '''
    Samples non-uniformly from an N-dimensional simplex projected onto an
    N-dimensional sphere using a stereographic projection.

    (Note that a 'stereographic spherical simplex' is _not_ the same thing as a
    spherical simplex defined using the submanifold geometry the N-dimensional sphere
    inherets from (N + 1)-dimensional Euclidean space.)

    Parameters
    ----------
    n_dim : int, optional, default: 3
        dimension of the sphere the points are sampled from (note that the
        sphere is a 2-dimensional space)

    simplex_radius : float, optional, default: 1
        radius of the simplex in Euclidean space, before it is projected onto
        the sphere

    n_samples : int, optional, default: 1000
        number of points sampled from the spherical simplex

    Returns
    -------
    A numpy array from shape (n_samples, n_dim + 1)
    '''

    X_hyperplane = _sample_from_n_simplex(n_dim + 1, simplex_radius, n_samples)
    print(X_hyperplane.shape)
    X_sphere = _stereographic_inverse(X_hyperplane)
    print(X_sphere.shape)

    return X_sphere

def train_epoch(model, data_loader):
    loss = 0
    reconstruction_loss = 0
    archetypal_loss = 0
    for batch_features, _ in data_loader:
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        batch_features = batch_features.view(-1, 784).to(device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        output, archetypal_embedding = model(batch_features)

        # compute training reconstruction loss
        curr_reconstruction_loss = torch.mean((output - batch_features)**2)
        reconstruction_loss += curr_reconstruction_loss

        # compute training archetypal loss
        curr_archetypal_loss = model.calc_archetypal_loss(archetypal_embedding)
        archetypal_loss += curr_archetypal_loss

        train_loss = curr_reconstruction_loss + curr_archetypal_loss

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(data_loader)
    reconstruction_loss = reconstruction_loss / len(data_loader)
    archetypal_loss = archetypal_loss / len(data_loader)
    return loss, reconstruction_loss, archetypal_loss
