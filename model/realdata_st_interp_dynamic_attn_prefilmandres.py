"""
Spatio-Temporal Interpolation Model

Advanced ST-DeepKriging Model (Ultimate Stable Version):
- Spatial Embedding: φ(s) via Wendland/Gaussian RBF
- Temporal Embedding: ψ(t) via Gaussian RBF
- Feature Fusion: Feature-wise Linear Modulation (FiLM)
- Backbone: Stem Layer + Residual Multi-Layer Perceptron (ResMLP)
"""
import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.mixture import GaussianMixture


class SpatialBasisEmbedding(nn.Module):
    # [原有的完整基函数代码，保持不变]
    CALIBRATION_FACTORS = {
        'wendland': 1.000000,
        'gaussian': 0.223477,
        'triangular': 0.654714,
    }

    def __init__(self, n_centers: list = [25, 81, 121], learnable: bool = False,
                 init_method: str = 'uniform', train_coords: np.ndarray = None,
                 basis_function: str = 'wendland', gradient_damping: bool = False,
                 damping_threshold: float = 0.3, damping_strength: float = 1.0):
        super().__init__()
        self.n_centers = n_centers
        self.learnable = learnable
        self.init_method = init_method
        self.basis_function = basis_function
        self.gradient_damping = gradient_damping
        self.damping_threshold = damping_threshold
        self.damping_strength = damping_strength

        if basis_function not in self.CALIBRATION_FACTORS:
            raise ValueError(f"Unknown basis function: {basis_function}")

        if init_method == 'uniform':
            centers, bandwidths = self._init_uniform()
        elif init_method == 'gmm':
            assert train_coords is not None, "train_coords required for GMM initialization"
            centers, bandwidths = self._init_gmm(train_coords)
        elif init_method == 'random_site':
            assert train_coords is not None, "train_coords required for random_site initialization"
            centers, bandwidths = self._init_random_site(train_coords)
        elif init_method == 'kmeans_balanced':
            assert train_coords is not None, "train_coords required for kmeans_balanced initialization"
            centers, bandwidths = self._init_kmeans_balanced(train_coords)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        if learnable:
            self.centers = nn.Parameter(centers)
            self.register_buffer('centers_init', centers.clone())
            self.log_bandwidths = nn.Parameter(torch.log(bandwidths))
            if gradient_damping:
                self.centers.register_hook(self._gradient_damping_hook)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('_bandwidths', bandwidths)

        self.k = centers.shape[0]

    def _gradient_damping_hook(self, grad):
        with torch.no_grad():
            movement = self.centers - self.centers_init
            distances = torch.norm(movement, dim=1, keepdim=True)
            excess_distance = torch.clamp(distances - self.damping_threshold, min=0.0)
            damping_factor = torch.exp(-self.damping_strength * excess_distance)
            damped_grad = grad * damping_factor
        return damped_grad

    @property
    def bandwidths(self):
        if self.learnable:
            return torch.exp(self.log_bandwidths)
        else:
            return self._bandwidths

    def _init_uniform(self):
        centers_list = []
        bandwidths_list = []
        for k in self.n_centers:
            side = int(math.sqrt(k))
            assert side * side == k, f"n_centers must be perfect squares, got {k}"
            x = torch.linspace(0, 1, side)
            y = torch.linspace(0, 1, side)
            try:
                xx, yy = torch.meshgrid(x, y, indexing='ij')
            except TypeError:
                xx, yy = torch.meshgrid(x, y)
            centers = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
            spacing = 1.0 / (side - 1) if side > 1 else 1.0
            bandwidth = 2.5 * spacing
            centers_list.append(centers)
            bandwidths_list.append(torch.full((k,), bandwidth))
        centers = torch.cat(centers_list, dim=0)
        bandwidths = torch.cat(bandwidths_list, dim=0)
        return centers, bandwidths

    def _init_gmm(self, train_coords: np.ndarray):
        centers_list = []
        bandwidths_list = []
        max_samples_for_gmm = 10000
        if len(train_coords) > max_samples_for_gmm:
            indices = np.random.choice(len(train_coords), max_samples_for_gmm, replace=False)
            train_coords_sub = train_coords[indices]
        else:
            train_coords_sub = train_coords

        uniform_bandwidths = []
        for k in self.n_centers:
            side = int(math.sqrt(k))
            spacing = 1.0 / (side - 1) if side > 1 else 1.0
            uniform_bw = 2.5 * spacing
            uniform_bandwidths.append(uniform_bw)

        train_coords_64 = train_coords_sub.astype(np.float64)

        for i, n_components in enumerate(self.n_centers):
            gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=42, max_iter=100,
                                  n_init=3, init_params='k-means++', reg_covar=1e-6, tol=1e-3, verbose=0)
            gmm.fit(train_coords_64)
            centers = torch.from_numpy(gmm.means_).float()
            stds = np.sqrt(gmm.covariances_)
            bandwidths_raw = 4.23 * 2.5 * stds
            uniform_bw = uniform_bandwidths[i]
            bw_min = 0.25 * uniform_bw
            bw_max = float('inf')
            bandwidths_clipped = np.clip(bandwidths_raw, bw_min, bw_max)
            bandwidths = torch.from_numpy(bandwidths_clipped).float()
            centers_list.append(centers)
            bandwidths_list.append(bandwidths)

        centers = torch.cat(centers_list, dim=0)
        bandwidths = torch.cat(bandwidths_list, dim=0)
        return centers, bandwidths

    def _init_random_site(self, train_coords: np.ndarray):
        from scipy.spatial.distance import cdist
        centers_list = []
        bandwidths_list = []
        for k in self.n_centers:
            if k > len(train_coords):
                indices = np.random.choice(len(train_coords), k, replace=True)
            else:
                indices = np.random.choice(len(train_coords), k, replace=False)
            centers = train_coords[indices]
            distances = cdist(centers, centers)
            np.fill_diagonal(distances, np.inf)
            n_neighbors = min(4, k - 1) if k > 1 else 1
            sorted_distances = np.sort(distances, axis=1)
            nearest_distances = sorted_distances[:, :n_neighbors]
            avg_distances = nearest_distances.mean(axis=1)
            bandwidth_scale = 2.5
            bandwidths = avg_distances * bandwidth_scale
            if k == 1:
                side = int(math.sqrt(self.n_centers[0]))
                spacing = 1.0 / (side - 1) if side > 1 else 1.0
                bandwidths = np.array([2.5 * spacing])
            centers_list.append(torch.from_numpy(centers).float())
            bandwidths_list.append(torch.from_numpy(bandwidths).float())

        centers = torch.cat(centers_list, dim=0)
        bandwidths = torch.cat(bandwidths_list, dim=0)
        return centers, bandwidths

    def _init_kmeans_balanced(self, train_coords: np.ndarray):
        from scipy.spatial.distance import cdist
        from k_means_constrained import KMeansConstrained
        centers_list = []
        bandwidths_list = []
        max_samples_for_kmeans = 10000
        if len(train_coords) > max_samples_for_kmeans:
            indices = np.random.choice(len(train_coords), max_samples_for_kmeans, replace=False)
            train_coords_sub = train_coords[indices]
        else:
            train_coords_sub = train_coords

        for k in self.n_centers:
            n_samples = len(train_coords_sub)
            size_per_cluster = n_samples // k
            size_min = max(1, size_per_cluster - 1)
            size_max = size_per_cluster + (n_samples % k)
            kmeans_balanced = KMeansConstrained(n_clusters=k, size_min=size_min, size_max=size_max, random_state=42,
                                                n_init=3, max_iter=100)
            kmeans_balanced.fit(train_coords_sub)
            centers = kmeans_balanced.cluster_centers_
            distances = cdist(centers, centers)
            np.fill_diagonal(distances, np.inf)
            n_neighbors = min(4, k - 1) if k > 1 else 1
            sorted_distances = np.sort(distances, axis=1)
            nearest_distances = sorted_distances[:, :n_neighbors]
            avg_distances = nearest_distances.mean(axis=1)
            bandwidth_scale = 2.5
            bandwidths = avg_distances * bandwidth_scale
            if k == 1:
                side = int(math.sqrt(self.n_centers[0]))
                spacing = 1.0 / (side - 1) if side > 1 else 1.0
                bandwidths = np.array([2.5 * spacing])
            centers_list.append(torch.from_numpy(centers).float())
            bandwidths_list.append(torch.from_numpy(bandwidths).float())

        centers = torch.cat(centers_list, dim=0)
        bandwidths = torch.cat(bandwidths_list, dim=0)
        return centers, bandwidths

    def forward(self, coords: torch.Tensor):
        dist = torch.cdist(coords.unsqueeze(0) if coords.dim() == 2 else coords, self.centers.unsqueeze(0))
        if coords.dim() == 2:
            dist = dist.squeeze(0)
        calibration = self.CALIBRATION_FACTORS[self.basis_function]
        r = dist / (self.bandwidths * calibration)

        if self.basis_function == 'wendland':
            phi = self._wendland(r)
        elif self.basis_function == 'gaussian':
            phi = self._gaussian(r)
        elif self.basis_function == 'triangular':
            phi = self._triangular(r)
        return phi

    def _wendland(self, r: torch.Tensor) -> torch.Tensor:
        r = r.clamp(max=1.0)
        return torch.pow(1 - r, 6) * (35 * r ** 2 + 18 * r + 3) / 3

    def _gaussian(self, r: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * r ** 2)

    def _triangular(self, r: torch.Tensor) -> torch.Tensor:
        return torch.clamp(1 - r, min=0.0)

    def compute_domain_penalty(self, domain_bounds=(0.0, 1.0)):
        if not self.learnable:
            return torch.tensor(0.0, device=self.centers.device)
        min_bound, max_bound = domain_bounds
        lower_violations = torch.clamp(min_bound - self.centers, min=0.0)
        upper_violations = torch.clamp(self.centers - max_bound, min=0.0)
        violations = lower_violations + upper_violations
        penalty = torch.sum(violations ** 2)
        return penalty

    def compute_movement_penalty(self):
        if not self.learnable:
            return torch.tensor(0.0, device=self.centers.device)
        movement = self.centers - self.centers_init
        penalty = torch.sum(movement ** 2)
        return penalty


class TemporalBasisEmbedding(nn.Module):
    """
    Advanced Temporal Basis: Gaussian RBF + Fourier Periodic Embedding

    Gaussian RBF plus two configurable physical Fourier periods.
    The default periods reproduce the original 2024 hourly PM2.5 model.
    """

    def __init__(self, n_centers: list = [10, 15, 45], T_total: int = 8784,
                 short_period: float = 24.0, long_period: float = 8784.0,
                 time_offset_steps: int = 0):
        super().__init__()
        self.n_centers = n_centers
        self.T_total = int(T_total)
        self.short_period = float(short_period)
        self.long_period = float(long_period)
        self.time_offset_steps = int(time_offset_steps)
        if self.T_total < 1 or self.short_period <= 0 or self.long_period <= 0:
            raise ValueError("T_total and Fourier periods must be positive")

        # 1. 高斯 RBF
        centers_list = []
        bandwidths_list = []
        for n in n_centers:
            centers = torch.linspace(0.0, 1.0, n)
            centers_list.append(centers)
            grid_spacing = 1.0 / (n - 1) if n > 1 else 1.0
            bandwidth = 2.5 * grid_spacing
            bandwidths_list.append(torch.full((n,), bandwidth))

        self.register_buffer('centers', torch.cat(centers_list))
        self.register_buffer('bandwidths', torch.cat(bandwidths_list))
        self.k_rbf = self.centers.shape[0]

        # 2. Two physical cycles (2 sine/cosine pairs, preserving feature count).
        self.k_periodic = 4
        self.k_time = self.k_rbf + self.k_periodic

    def forward(self, t: torch.Tensor):
        """
        t: (B, 1) - 归一化的时间 t_normalized in [0, 1]
        """
        # A. 计算 RBF 特征
        diff = t - self.centers.view(1, -1)
        scaled_diff = diff / self.bandwidths.view(1, -1)
        psi_rbf = torch.exp(-0.5 * scaled_diff ** 2)

        # B. 计算物理周期特征
        # Recover original sampling steps after windowing and normalization.
        t_real = t * (self.T_total - 1) + self.time_offset_steps

        # Hourly data: 24-hour cycle; daily GSOD: configurable monthly scale.
        omega_day = 2 * math.pi / self.short_period
        sin_day = torch.sin(omega_day * t_real)
        cos_day = torch.cos(omega_day * t_real)

        # Physical annual cycle, independent of the selected window length.
        omega_year = 2 * math.pi / self.long_period
        sin_year = torch.sin(omega_year * t_real)
        cos_year = torch.cos(omega_year * t_real)

        psi_periodic = torch.cat([sin_day, cos_day, sin_year, cos_year], dim=-1)

        # C. 拼接输出 [高斯特征, 周期特征]
        psi_t = torch.cat([psi_rbf, psi_periodic], dim=-1)
        return psi_t


# =================================================================
# [新增] 带有残差连接的 MLP 模块 (Residual MLP Block)
# =================================================================
class ResMLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, layernorm=True):
        super().__init__()
        # Pre-activation: 先 LayerNorm，再 ReLU，最后 Linear
        # 这保证了 Skip 路径绝对不会受到非线性的污染
        self.ln = nn.LayerNorm(in_dim) if layernorm else nn.Identity()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout)

        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        # 纯正的 Pre-Activation 残差流
        out = self.ln(x)
        out = self.relu(out)
        out = self.linear(out)
        out = self.dropout(out)
        return out + self.skip(x)


class STInterpMLP(nn.Module):
    """
    Spatio-Temporal Interpolation via FiLM Modulation and ResMLP Backbone
    """

    def __init__(self,
                 p: int = 0,
                 k_spatial_centers: list = [25, 81, 121],
                 k_temporal_centers: list = [10, 15, 45],
                 hidden_dims: list = [256, 256, 128],
                 dropout: float = 0.1,
                 layernorm: bool = True,
                 spatial_learnable: bool = False,
                 spatial_init_method: str = 'uniform',
                 spatial_basis_function: str = 'wendland',
                 train_coords: np.ndarray = None,
                 gradient_damping: bool = False,
                 damping_threshold: float = 0.3,
                 damping_strength: float = 1.0,
                 T_total: int = 8784,
                 short_period: float = 24.0,
                 long_period: float = 8784.0,
                 time_offset_steps: int = 0,
                 output_dim: int = 1):
        super().__init__()

        self.p = p
        self.k_spatial_centers = k_spatial_centers
        self.spatial_init_method = spatial_init_method
        self.spatial_basis_function = spatial_basis_function
        self.output_dim = output_dim

        # Spatial basis embedding
        self.spatial_basis = SpatialBasisEmbedding(
            n_centers=k_spatial_centers,
            learnable=spatial_learnable,
            init_method=spatial_init_method,
            train_coords=train_coords,
            basis_function=spatial_basis_function,
            gradient_damping=gradient_damping,
            damping_threshold=damping_threshold,
            damping_strength=damping_strength
        )

        # Temporal basis embedding
        self.temporal_basis = TemporalBasisEmbedding(
            n_centers=k_temporal_centers,
            T_total=T_total,
            short_period=short_period,
            long_period=long_period,
            time_offset_steps=time_offset_steps,
        )
        self.k_spatial = self.spatial_basis.k
        self.k_temporal = self.temporal_basis.k_time

        # =================================================================
        # [融合核心 1]：FiLM 特征线性调制网络
        # =================================================================
        self.attn_hidden_dim = 96
        # self.attn_hidden_dim = 128
        # self.time_to_space_film = nn.Sequential(
        #     nn.Linear(self.k_temporal, self.attn_hidden_dim),
        #     nn.LayerNorm(self.attn_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.attn_hidden_dim, self.k_spatial * 2)  # 输出 Gamma 和 Beta
        # )

        # 在 __init__ 中找到原来的 time_to_space_film
        # 改成：
        self.context_to_space_film = nn.Sequential(
            # 输入维度变成了：时间基底数量 + 气象协变量数量
            nn.Linear(self.k_temporal + self.p, self.attn_hidden_dim),
            nn.LayerNorm(self.attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.attn_hidden_dim, self.k_spatial * 2)  # 输出 Gamma 和 Beta
        )

        # 零初始化策略：防止初期训练崩溃
        nn.init.zeros_(self.context_to_space_film[3].weight)
        nn.init.zeros_(self.context_to_space_film[3].bias)
        # 将 Gamma 的初始偏置设为 1.0，Beta 保持 0.0 (此时调制相当于恒等映射)
        self.context_to_space_film[3].bias.data[:self.k_spatial] = 1.0

        self.last_attn_weights = None
        # 计算拼接后的输入维度
        input_dim = p + self.k_spatial + self.k_temporal + self.k_spatial

        # =================================================================
        # [融合核心 2]：Stem Layer + ResMLP 残差特征融合网络
        # =================================================================
        mlp_dropout_prob = dropout
        self.input_dropout = nn.Dropout(p=mlp_dropout_prob)

        layers = []

        # 【极其重要修复】：引入独立的 Stem Layer 作为网络的大门
        # 作用 1：将拼接后的高维特征降维到 hidden_dims[0]
        # 作用 2：不带残差捷径，防止过拟合特征绕过 L1 稀疏惩罚
        stem_dim = hidden_dims[0]
        self.stem_linear = nn.Linear(input_dim, stem_dim)
        layers.append(self.stem_linear)

        if layernorm:
            layers.append(nn.LayerNorm(stem_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=mlp_dropout_prob))

        # 正式开启带有残差连接的高速网络
        # 注意：因为第一层已经由 stem_linear 承担，这里从 hidden_dims 的第二个元素开始循环
        prev_dim = stem_dim
        for hidden_dim in hidden_dims[1:]:
            layers.append(ResMLPBlock(
                in_dim=prev_dim,
                out_dim=hidden_dim,
                dropout=mlp_dropout_prob,
                layernorm=layernorm
            ))
            prev_dim = hidden_dim

        # 最后一层映射到目标回归维度 (不使用残差)
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.mlp = nn.Sequential(*layers)

    def compute_domain_penalty(self):
        return self.spatial_basis.compute_domain_penalty()

    def compute_movement_penalty(self):
        return self.spatial_basis.compute_movement_penalty()

    def compute_sparsity_penalty(self, penalty_type='element', lambda_l1=0.01, lambda_group=0.01):
        if penalty_type not in ['element', 'group', 'sparse_group', 'none']:
            raise ValueError(f"Unknown penalty_type: {penalty_type}")

        if penalty_type == 'none':
            return {
                'spatial_penalty': torch.tensor(0.0, device=next(self.parameters()).device),
                'temporal_penalty': torch.tensor(0.0, device=next(self.parameters()).device),
                'total_penalty': torch.tensor(0.0, device=next(self.parameters()).device)
            }

        # 【极其重要修复】：直接提取无捷径的 Stem Layer 权重，堵死作弊通道
        weight = self.stem_linear.weight

        idx = 0
        if self.p > 0:
            idx += self.p

        spatial_weight = weight[:, idx:idx + self.k_spatial].T
        idx += self.k_spatial
        temporal_weight = weight[:, idx:idx + self.k_temporal].T

        spatial_penalty = self._compute_penalty_for_block(spatial_weight, penalty_type, lambda_l1, lambda_group)
        temporal_penalty = self._compute_penalty_for_block(temporal_weight, penalty_type, lambda_l1, lambda_group)

        return {
            'spatial_penalty': spatial_penalty,
            'temporal_penalty': temporal_penalty,
            'total_penalty': spatial_penalty + temporal_penalty
        }

    def _compute_penalty_for_block(self, weight_block, penalty_type, lambda_l1, lambda_group):
        if penalty_type == 'element':
            return lambda_l1 * weight_block.abs().sum()
        elif penalty_type == 'group':
            penalty = torch.linalg.vector_norm(weight_block, dim=1).sum()
            return lambda_group * penalty
        elif penalty_type == 'sparse_group':
            group_penalty = torch.linalg.vector_norm(weight_block, dim=1).sum()
            element_penalty = weight_block.abs().sum()
            return lambda_group * group_penalty + lambda_l1 * element_penalty
        else:
            return torch.tensor(0.0, device=weight_block.device)

    # ---------------------------------------------------------------
    # 前向传播：执行 FiLM 调制 + Stem 融合 + ResMLP 推理
    # ---------------------------------------------------------------
    # def forward(self, X: torch.Tensor, coords: torch.Tensor, t: torch.Tensor):
    #     # 1. 提取原始时空特征
    #     phi_s = self.spatial_basis(coords)  # (B, k_spatial)
    #     psi_t = self.temporal_basis(t)  # (B, k_temporal)
    #
    #     # 2. 生成 FiLM 调制参数
    #     film_params = self.time_to_space_film(psi_t)
    #     gamma = film_params[:, :self.k_spatial]  # (B, k_spatial)
    #     beta = film_params[:, self.k_spatial:]  # (B, k_spatial)
    #
    #     # 记录 Gamma 权重用于热力图可视化
    #     self.last_attn_weights = gamma.detach()
    #
    #     # 执行 FiLM 特征线性调制
    #     dynamic_phi_s = phi_s * gamma + beta
    #
    #     # 3. 拼接特征
    #     if X is not None and X.numel() > 0 and self.p > 0:
    #         features = torch.cat([X, phi_s, psi_t, dynamic_phi_s], dim=-1)
    #     else:
    #         features = torch.cat([phi_s, psi_t, dynamic_phi_s], dim=-1)
    #
    #     # 4. 进入后续融合网络推理
    #     features = self.input_dropout(features)
    #     y_pred = self.mlp(features)
    #
    #     return y_pred

    # ---------------------------------------------------------------
    # 前向传播：执行 FiLM 调制 + Stem 融合 + ResMLP 推理 (终极防御版)
    # ---------------------------------------------------------------
    def forward(self, X: torch.Tensor, coords: torch.Tensor, t: torch.Tensor):
        # 1. 提取原始时空特征
        phi_s = self.spatial_basis(coords)  # (B, k_spatial)
        psi_t = self.temporal_basis(t)  # (B, k_temporal)

        # =========================================================
        # 【防御机制 1】：确保气象特征 X 永远存在，维度永远正确
        # =========================================================
        if self.p > 0:
            # 如果外界画图函数忘记传 X，或者传了个空的进来，自动补全 0
            # (在 Z-Score 体系中，0 代表该气象要素的平均状态，完美契合画图需求)
            if X is None or X.numel() == 0 or X.shape[-1] != self.p:
                X = torch.zeros(phi_s.shape[0], self.p, device=phi_s.device, dtype=phi_s.dtype)

            # Context = [时间, 气象] -> 让时间和气象共同决定基站生死
            context = torch.cat([psi_t, X], dim=-1)
        else:
            context = psi_t

        # 2. 生成 FiLM 调制参数
        # (确保你的 __init__ 里的 context_to_space_film 已经改为了 k_temporal + self.p)
        film_params = self.context_to_space_film(context)
        gamma = film_params[:, :self.k_spatial]  # (B, k_spatial)
        beta = film_params[:, self.k_spatial:]  # (B, k_spatial)

        # 记录 Gamma 权重用于热力图可视化
        self.last_attn_weights = gamma.detach()

        # 执行 FiLM 特征线性调制
        dynamic_phi_s = phi_s * gamma + beta

        # =========================================================
        # 【防御机制 2】：无脑拼接，彻底封死维度错乱的可能
        # =========================================================
        if self.p > 0:
            features = torch.cat([X, phi_s, psi_t, dynamic_phi_s], dim=-1)
        else:
            features = torch.cat([phi_s, psi_t, dynamic_phi_s], dim=-1)

        # 4. 进入后续融合网络推理
        features = self.input_dropout(features)
        y_pred = self.mlp(features)

        return y_pred


def create_model(config: dict, train_coords: np.ndarray = None) -> STInterpMLP:
    """
    Create model from config
    """
    regression_type = config.get('regression_type', 'mean')
    if regression_type == 'multi-quantile':
        quantile_levels = config.get('quantile_levels', [0.1, 0.5, 0.9])
        output_dim = len(quantile_levels)
    else:
        output_dim = 1

    return STInterpMLP(
        p=config.get('p_covariates', 0),
        k_spatial_centers=config.get('k_spatial_centers', [25, 81, 121]),
        k_temporal_centers=config.get('k_temporal_centers', [10, 15, 45]),
        hidden_dims=config.get('hidden_dims', [256, 256, 128]),
        dropout=config.get('dropout', 0.1),
        layernorm=config.get('layernorm', True),
        spatial_learnable=config.get('spatial_learnable', False),
        spatial_init_method=config.get('spatial_init_method', 'uniform'),
        spatial_basis_function=config.get('spatial_basis_function', 'wendland'),
        train_coords=train_coords,
        gradient_damping=config.get('gradient_damping', False),
        damping_threshold=config.get('damping_threshold', 0.3),
        damping_strength=config.get('damping_strength', 1.0),
        T_total=config.get('T_total', 8784),
        short_period=config.get('short_period', 24.0),
        long_period=config.get('long_period', 8784.0),
        time_offset_steps=config.get('time_offset_steps', 0),
        output_dim=output_dim
    )
