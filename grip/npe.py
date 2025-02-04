#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import os
from sbi.utils import posterior_nn # The purpose of this import is to avoids a circular import in `from sbi.neural_nets.net_builders import build_maf`
from sbi.neural_nets.net_builders import build_maf


class GripNPE(object):
    """
    This class is a framework to use the Neural Posterior Estimation technique
    with GRIP. 
    It relies on the libraries `LAMPE <https://github.com/probabilists/lampe>`_ and
    `PyTorch <https://pytorch.org/>`_ (thus it can run on GPU if desired).

    The idea is to simulate data and train the neural network to map these data with some parameters.
    During inference, real data are fed to the neural network to infer the posterior of the parameters.
    
    The loss function is the negative log-likelihood of the NPE normalising flow (see LAMPE documentation).

    Parameters
    ----------
    wl_scale : 1d-array
        Wavelength scale.
    func_model : function or callable
        Function formatting the data to be compared to a model (e.g. histogram or power density spectrum).
    func_args : list, optional
        Arguments used or transmitted by `func_model`. The default is ().
    func_kwargs : dict, optional
        Dictionary of keywords used or transmitted by `func_model`. The default is {}.
    use_cuda : bool, optional
        Use cuda for training the neural network or inferring, providing CUDA is available. The default is True.

    Returns
    -------
    None.
    
    """
    def __init__(self, wl_scale, func_model, func_args=(), func_kwargs={}, use_cuda=True):
        """
        

        Parameters
        ----------
        wl_scale : 1d-array
            Wavelength scale.
        func_model : function or callable
            Function formatting the data to be compared to a model (e.g. histogram or power density spectrum).
        func_args : list, optional
            Arguments used or transmitted by `func_model`. The default is ().
        func_kwargs : dict, optional
            Dictionary of keywords used or transmitted by `func_model`. The default is {}.
        use_cuda : bool, optional
            Use cuda for training the neural network or inferring, providing CUDA is available. The default is True.

        Returns
        -------
        None.

        """
        self.wl_scale = wl_scale
        self.func_model = func_model
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.loss = None
        self.maf_estimator = None
        
        
        if use_cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
    def draw_params(self, dist_type, dist_params_theta):
        """
        Generate parameters sets to simulate data. These parameters are the one
        to be inferred by the NN.

        Parameters
        ----------
        dist_type : list of strings
            Define the distribution to choose for each parameter. 
            Available keywords are *normal* (for normal distribution) or *uniform* (for uniform distribution).
            If the keyword is not recognised, a uniform distribution is considered.
        dist_params_theta : List of tuples
            List of parameters to use to generate random values from a distribution defined by `dist_type`.

        Returns
        -------
        params : array-like
            Parameters generated.

        Examples
        --------
        >>> dist_type = ['uniform', 'normal', 'uniform']
        >>> dist_params_theta = [(0, 0.01), (300, 100), (100, 20)]
        >>> draw_params(dist_type, dist_params_theta)
        """
        params = []
        for i in range(len(dist_params_theta)):
            p = dist_params_theta[i]
            if dist_type[i] == 'normal':
                param = np.random.normal(*p)
            else:
                param = np.random.uniform(*p)
            params.append(param)
        
        params = np.array(params)
        return params
    
    def simulator(self, params):
        """
        Simulate the data to train the neural network.

        Parameters
        ----------
        params : array-like
            Parameters to infer of the model to fit on the data.

        Returns
        -------
        out : 1d-array
            Modelled data.

        """
        out = self.func_model(params, *self.func_args, **self.func_kwargs)[0]
        return out
    
    def generate_batch(self, batch_size, dist_type, dist_params_theta):
        """
        Generate batch of data to train de neural network and their associated parameters.

        Parameters
        ----------
        batch_size : int
            Number of simulated data sequence in a batch.
        dist_type : list of strings
            Define the distribution to choose for each parameter. 
            Available keywords are *normal* (for normal distribution) or *uniform* (for uniform distribution).
        dist_params_theta : List of tuples
            List of parameters to use to generate random values from a distribution defined by `dist_type`.

        Returns
        -------
        batched_data : array
            Batched simulated data of shape (batch_size, \*data.shape).
        batched_theta : array
            Batched parameters to infer of shape (batch_size, number of parameters).

        """
        batched_data = []
        batched_theta = []
        
        for i in range(batch_size):
            theta = self.draw_params(dist_type, dist_params_theta)
            func_out = self.simulator(theta)
            batched_data.append(func_out)
            batched_theta.append(theta)
            
        return  batched_data, batched_theta
    
    def create_train_set(self, nb_batch_per_epoch, batch_size, dist_type, dist_params_theta):
        """
        Generate data set to train the neural network.
        It creates several batched data to use in one training epoch.

        Parameters
        ----------
        nb_batch_per_epoch : int
            Number of batches to train the neural network during one epoch.
        batch_size : int
            Number of simulated data sequence in a batch.
        dist_type : list of strings
            Define the distribution to choose for each parameter. 
            Available keywords are *normal* (for normal distribution) or *uniform* (for uniform distribution).
        dist_params_theta : List of tuples
            List of parameters to use to generate random values from a distribution defined by `dist_type`.

        Returns
        -------
        train_data : nd-array
            Dataset to train the neural network of shape `(nb_batch_per_epoch, batch_size, data_size)`.
        train_params : nd-array
            Associated parameters that have created `train_data` of shape `(nb_batch_per_epoch, batch_size, number_of_params)`.

        """
        train_data = []
        train_params = []

        for i in tqdm(range(nb_batch_per_epoch)):
            batched_data, batched_theta = self.generate_batch(batch_size, dist_type, dist_params_theta)
            train_data.append(batched_data)
            train_params.append(batched_theta)

        train_data = np.array(train_data)
        train_params = np.array(train_params)
        
        self.train_data = torch.from_numpy(train_data).to(torch.float32).to(self.device)
        self.train_params = torch.from_numpy(train_params).to(torch.float32).to(self.device)
        
        return train_data, train_params

    def set_nn(self, nb_transforms, nb_layers, hidden_features):
        """
        Create the normalising flow to infer data.
        The flow is a *masked autoregressive flow*.

        Parameters
        ----------
        nb_transforms : int
            Number of transformation in the normalising flow. It is used by the LAMPE library.
        nb_layers : int
            Number of layers in a Transform component.
        hidden_features : int
            Number of features out of each layer.

        Attributes
        ----------
        self.maf_estimator : neural network estimator flow

        """
        
        # The normalising flow requires the shape of a single sample of data or params
        # But it also assumes the input has shape [nb samples, *data shape]
        # self.train_params and self.train_data have shape [nb batches, nb sample, *data_shape]
        # So we need to crop as below:
        tr_params = self.train_params[0]
        tr_data = self.train_data[0]
        
        self.maf_estimator = build_maf(tr_params, tr_data, hidden_features=hidden_features, 
                                       num_transforms=nb_transforms, num_blocks=nb_layers)
        self.maf_estimator = self.maf_estimator.to(self.device)
        
    def train_nn(self, nb_epoch, show_plot=False, save_fig=False, save_path='cluster.png'):
        """
        Train the neural network following the Neural Posterior Estimation flow.

        Parameters
        ----------
        nb_epoch : int
            Number of epochs to train the neural network.
        show_plot : bool, optional
            Show the learning curve. The default is False.
        save_fig : bool, optional
            Save the learning curve. The default is False.
        save_path : string, optional
            Path to save the figure. This path must end by the name of the file and its extension. The default is 'cluster.png'. The figure is saved as a PNG with dpi=150.

        Returns
        -------
        training_loss : 1d-array
            Values of the learning curve. The size of the array is the number of epochs.

        """
        train_set = (self.train_params, self.train_data)
        
        training_loss = []
        self.optimizer = optim.AdamW(list(self.maf_estimator.parameters()), lr=1e-3, weight_decay=1e-2)

        with tqdm(range(nb_epoch), unit='epoch') as tq:
            for ep in tq:
                for (batch_param, batch_data) in zip(*train_set):
                    self.optimizer.zero_grad()
                    losses = self.maf_estimator.loss(batch_param, condition=batch_data)
                    loss = torch.mean(losses)
                    loss.backward()
                    
                    self.optimizer.step()
                
                training_loss.append(loss.item())
                tq.set_postfix(loss=loss.item())
                
        if show_plot:
            plt.figure(figsize=(4, 3))
            plt.semilogy(training_loss, label='loss')
            plt.legend(loc='best')
            plt.title('Evolution of the loss function')
            if save_fig:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)            
                plt.savefig(save_path, format='png', dpi=150)
                
        return training_loss

   
    def inference_on_data(self, data_to_inf, nb_sample):
        """
        Infer data to deduce the posterior of the parameters with the trained neural network.

        Parameters
        ----------
        data_to_inf : nd-array
            Data to infer. They must be in the same shape and format as the simulated data used to train the neural network.
        nb_sample: int
            Number of samples to generate from the surrogate posteriors of the parameters.

        Returns
        -------
        samples : nd-array
            Samples of the inferred parameters, which distributions are their posteriors. Shape is (nb elements, nb parameters).

        """
        data_inf = torch.from_numpy(data_to_inf).to(torch.float32).to(self.device)
        
        with torch.no_grad():
            samples = self.maf_estimator.sample((nb_sample,), condition=data_inf)
        self.samples = samples.detach()
        self.samples = self.samples.cpu().numpy()
            
        return self.samples