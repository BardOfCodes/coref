
import torch as th
import torch.nn as nn
from .features_extractor_3d import Vox3DCNN
from .features_extractor_2d import Vox2DCNN
from .vpi_net import BaseVPINet


class Sampler(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, hidden_size)
        self.mlp2var = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        encode = th.relu(self.mlp1(x))

        mu = self.mlp2mu(encode)
        logvar = self.mlp2var(encode)

        return mu, logvar


class Vox3DVAE(Vox3DCNN):
    """ Attach a VAE head to the input code.
    """

    def __init__(self, features_dim, dropout=0.0, first_stride=2, out_len=64, latent_dim=128):

        super(Vox3DVAE, self).__init__(
            features_dim, dropout, first_stride, out_len)

        self.latent_dim = latent_dim
        self.sampler = Sampler(out_len * features_dim, latent_dim)
        self.up_ll = nn.Linear(latent_dim, out_len * features_dim)
        self.mode = "encode"

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, input):
        batch_size = input.shape[0]
        features = super(Vox3DVAE, self).forward(input)
        features = features.view(batch_size, -1)
        mu, logvar = self.sampler(features)
        std = logvar.mul(0.5).exp_()
        eps = th.randn_like(std)

        gaussian_code = eps.mul(std).add_(mu)

        features = self.up_ll(gaussian_code)
        features = features.view(batch_size, self.out_len, -1)
        return features

    def forward_train(self, input):
        batch_size = input.shape[0]
        features = super(Vox3DVAE, self).forward(input)
        features = features.view(batch_size, -1)
        mu, logvar = self.sampler(features)
        std = logvar.mul(0.5).exp_()
        eps = th.randn_like(std)

        gaussian_code = eps.mul(std).add_(mu)

        features = self.up_ll(gaussian_code)
        features = features.view(batch_size, self.out_len, -1)
        return features, mu, logvar


class Vox2DVAE(Vox2DCNN):
    """ Attach a VAE head to the input code.
    """

    def __init__(self, features_dim, dropout=0.0, first_stride=2, out_len=64, latent_dim=128):

        super(Vox2DVAE, self).__init__(
            features_dim, dropout, first_stride, out_len)

        self.latent_dim = latent_dim
        self.sampler = Sampler(out_len * features_dim, latent_dim)
        self.up_ll = nn.Linear(latent_dim, out_len * features_dim)
        self.mode = "encode"

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, input):
        batch_size = input.shape[0]
        features = super(Vox2DVAE, self).forward(input)
        features = features.view(batch_size, -1)
        mu, logvar = self.sampler(features)
        std = logvar.mul(0.5).exp_()
        eps = th.randn_like(std)

        gaussian_code = eps.mul(std).add_(mu)

        features = self.up_ll(gaussian_code)
        features = features.view(batch_size, self.out_len, -1)
        return features

    def forward_train(self, input):
        batch_size = input.shape[0]
        features = super(Vox2DVAE, self).forward(input)
        features = features.view(batch_size, -1)
        mu, logvar = self.sampler(features)
        std = logvar.mul(0.5).exp_()
        eps = th.randn_like(std)

        gaussian_code = eps.mul(std).add_(mu)

        features = self.up_ll(gaussian_code)
        features = features.view(batch_size, self.out_len, -1)
        return features, mu, logvar


class VPINetVAE(BaseVPINet):

    def set_feature_extractor(self, config):
        if config.n_dims == 3:
            cnn_class = Vox3DVAE
        elif config.n_dims == 2:
            cnn_class = Vox2DVAE
        self.cnn_extractor = cnn_class(self.hidden_dim, self.dropout,
                                       first_stride=self.cnn_first_stride,
                                       out_len=self.visual_seq_len,
                                       latent_dim=self.vae_latent_dim)

    def set_config(self, config):
        super(VPINetVAE, self).set_config(config)
        self.vae_latent_dim = config.vae_latent_dim

    def forward_train(self, batch_dict):

        x_in = batch_dict['occs']
        actions_in = batch_dict['actions']

        cnn_features, mu, logvar = self.cnn_extractor.forward_train(x_in)
        token_embeddings = self.token_embedding(actions_in)

        out = self.pos_encoding(
            th.cat((cnn_features, token_embeddings), dim=1))

        for attn_layer in self.attn_layers:
            out = attn_layer(out, self.attn_mask, None)
        # should be self.prog_seq_len
        seq_out = out[:, self.visual_seq_len:-1, :]

        output = self.stack_all_vectors(seq_out)

        cmd_distr = self.attn_to_output(output)
        cmd_logsoft = self.cmd_logsmax(cmd_distr)
        # cmd_distr = th.softmax(cmd_distr, dim = 1)
        return cmd_logsoft, mu, logvar

    def forward_beam_init(self, input_batch, beam_size):

        input_occ = input_batch["occs"]
        batch_size = input_occ.shape[0]
        start_token = self.token_embedding.num_embeddings - 2

        cnn_features = self.cnn_extractor.forward(input_occ)
        token_seq = th.zeros(batch_size, self.prog_seq_len,
                             device=self.device).long()
        token_seq[:, 0] = start_token

        embedding_seq = th.zeros(batch_size, self.max_length,
                                 self.token_embedding.embedding_dim,
                                 device=self.device)

        token_embedding = self.token_embedding(token_seq) * 0
        features = th.cat((cnn_features, token_embedding), dim=1)
        embedding_seq[:, :] = self.pos_encoding(features)

        embedding_seq = embedding_seq.unsqueeze(
            1).expand(-1, beam_size, -1, -1)

        return embedding_seq
