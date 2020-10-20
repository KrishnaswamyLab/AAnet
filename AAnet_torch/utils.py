import numpy as np
import torch
from sklearn.decomposition import PCA

def get_n_simplex(n=2, scale=1, device=None):
    '''
    Returns an n-simplex centered at the origin
    '''

    nth = 1/(n-1)*(1-np.sqrt(n)) * np.ones(n-1)
    D = np.vstack([np.eye(n-1), nth]) * scale
    return torch.tensor(D - np.mean(D, axis=0), device=device)


def generate_data_on_sphere(n_obs=1000, radius=1):
    '''
    Generates data on that lies on a sphere of `radius`.

    TODO: not convinced that this is the correct way to project points onto sphere
          also doesn't scale to more than 3 dimensions :(
    '''
    np.random.seed(0)

    n_obs = n_obs
    n_dim = 2
    n_archetypes = 3

    # Get coordinates of archetypes in R2
    archetypes = get_n_simplex(n=n_archetypes).numpy()

    # Uniformly sample from a simplex
    data = np.random.uniform(0, 1, size=[n_obs, n_archetypes])
    data = -np.log(data)
    data = data / np.sum(data, axis=1, keepdims=True)

    # Project the points to appear within the archetypal coordiantes
    data_atype = data @ archetypes

    # 2D points on simplex to points on a sphere
    data_sphere  = _latlong_to_xyz(data_atype, radius=radius)
    atype_sphere = _latlong_to_xyz(archetypes, radius=radius)

    # Center the data
    data_center = data_sphere.mean(axis=0, keepdims=True)
    data_sphere -= data_center
    atype_sphere -= data_center

    return data_sphere, atype_sphere


def _latlong_to_xyz(X, radius=1):
    '''
    Takes 2D input as lat/long coordinates and returns coordinates in R3
    '''
    longitude = X[:,0] / radius
    latitude = 2 * np.arctan(np.exp(X[:,1]/radius)) - np.pi/2
    x3 = radius * np.cos(latitude) * np.cos(longitude)
    y3 = radius * np.cos(latitude) * np.sin(longitude)
    z3 = radius * np.sin(latitude)
    #x3 = (x3 - R) / 10
    return np.vstack([x3, y3, z3]).T

def get_diffusion_extrema(data, diffusion_potential, n_archetypes, n_pcs=50):	
    pc_op = PCA(n_pcs)	
    data_pc = pc_op.fit_transform(diffusion_potential)	
    extrema_idx = []	
    for i in range(int(np.ceil(n_archetypes/2.0))):	
        extrema_idx.append(np.argmin(data_pc[:,i]))	
        extrema_idx.append(np.argmax(data_pc[:,i]))	
        	
    # if odd number of archetypes, use argmin only from last component
    extrema = data[extrema_idx[:n_archetypes]]	
    return extrema

def train_epoch(model, data_loader, optimizer, epoch, extrema=None,	
                gamma_reconstruction=1.0, gamma_archetypal=1.0, gamma_extrema=1.0):
    loss = 0
    reconstruction_loss = 0
    archetypal_loss = 0
    extrema_loss = 0
    
    for idx, data in enumerate(data_loader):	
        # if input is list, then data_loader contains features and target	
        # in this case, assume first input is features based on data_loader structure	
        if isinstance(data, list):	
            batch_features = data[0]	
        else:	
            batch_features = data	
        	
        # if not using diffusion extrema, extrema=None
        # else, add diffusion extrema to beginning of each extrema	
        # first n_archetypes samples are used in extrema loss	
        if extrema is not None:	
            batch_features = torch.cat((extrema.view(-1, model.input_shape),	
                                    batch_features.view(-1, model.input_shape)), 0)	
            	
        # reshape mini-batch data to [N, input_shape] matrix	
        batch_features = batch_features.view(-1, model.input_shape)	
        	
        # load it to the active device	
        batch_features = batch_features.to(model.device)

        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()

        # compute reconstructions
        output, archetypal_embedding = model(batch_features.float())

        # compute training reconstruction loss
        curr_reconstruction_loss = torch.mean((output - batch_features)**2)
        reconstruction_loss += curr_reconstruction_loss

        # compute training archetypal loss
        curr_archetypal_loss = model.calc_archetypal_loss(archetypal_embedding)
        archetypal_loss += curr_archetypal_loss
        
        # compute training diffusion extrema loss	
        if extrema is not None:	
            curr_extrema_loss = model.calc_diffusion_extrema_loss(archetypal_embedding)	
            extrema_loss += curr_extrema_loss	
        else:	
            curr_extrema_loss = 0	
            extrema_loss = 0	
            	
        # extrema penalization decreases over batches and epochs	
        # this enables AAnet to learn the correct archetypes if the diffusion extrema are not close	
        train_loss = gamma_reconstruction * curr_reconstruction_loss + \	
                     gamma_archetypal * curr_archetypal_loss + \	
                     gamma_extrema /(epoch * len(data_loader) + (idx+1)) * curr_extrema_loss

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
