import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import diff_CSDI
from utils.prepare4llm import get_llm


class ScaleRouter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.net(features)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size=25):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class CSDI_series_decomp(nn.Module):
    def __init__(self, lookback_len, pred_len, kernel_size=25):
        super(CSDI_series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.lookback_len = lookback_len
        self.pred_len = pred_len

    def forward(self, x):
        x = x.permute(0, 2, 1)
        lookback = x[:, :self.lookback_len, :]

        moving_mean = self.moving_avg(lookback)
        res = lookback - moving_mean
        
        moving_mean = moving_mean.permute(0, 2, 1)
        res = res.permute(0, 2, 1)

        moving_mean = nn.functional.pad(moving_mean, (0, self.pred_len), "constant", 0)
        res = nn.functional.pad(res, (0, self.pred_len), "constant", 0)
        return res, moving_mean
        

    

class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device, window_lens):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.timestep_branch = config["model"]["timestep_branch"]
        self.timestep_emb_cat = config["model"]["timestep_emb_cat"]
        self.with_texts = config["model"]["with_texts"]
        self.noise_esti = config["diffusion"]["noise_esti"]
        self.relative_size_emb_cat = config["model"]["relative_size_emb_cat"]
        self.decomp = config["model"]["decomp"]
        self.ddim = config["diffusion"]["ddim"]
        self.sample_steps = config["diffusion"]["sample_steps"]
        self.sample_method = config["diffusion"]["sample_method"]

        self.lookback_len = config["model"]["lookback_len"]
        self.pred_len = config["model"]["pred_len"]
        self.diff_channels = config["diffusion"]["channels"]
        self.cfg = config["diffusion"]["cfg"]
        self.trend_cfg = config["diffusion"].get("trend_cfg", False)
        self.trend_cfg_power = config["diffusion"].get("trend_cfg_power", 1.0)
        self.trend_cfg_random = config["diffusion"].get("trend_cfg_random", False)
        self.trend_strength_scale = config["diffusion"].get("trend_strength_scale", 1.0)
        self.trend_volatility_scale = config["diffusion"].get("trend_volatility_scale", 1.0)
        self.trend_time_floor = config["diffusion"].get("trend_time_floor", 0.0)
        self.self_condition = bool(config["diffusion"].get("self_condition", False))
        self.self_condition_prob = float(config["diffusion"].get("self_condition_prob", 0.5))
        self.self_condition_target_only = bool(config["diffusion"].get("self_condition_target_only", True))
        self.c_mask_prob = config["diffusion"]["c_mask_prob"]
        self.context_dim = config["model"]["context_dim"]
        self.llm = config["model"]["llm"]
        self.domain = config["model"]["domain"]
        self.save_attn = config["model"]["save_attn"]
        self.save_token = config["model"]["save_token"]
        self.use_text_score_gate = bool(config["model"].get("use_text_score_gate", False))
        self.text_score_gate_strength = float(config["model"].get("text_score_gate_strength", 1.0))
        self.text_score_gate_floor = float(config["model"].get("text_score_gate_floor", 0.0))
        self.text_score_model_path = config["model"].get("text_score_model_path")
        self.text_score_model = self._load_text_score_model(self.text_score_model_path)
        train_cfg = config.get("train", {})
        self.multi_res_loss_weight = float(train_cfg.get("multi_res_loss_weight", 0.0))
        self.multi_res_use_huber = bool(train_cfg.get("multi_res_use_huber", True))
        self.multi_res_huber_delta = float(train_cfg.get("multi_res_huber_delta", 1.0))
        default_multi_res_mode = "static_band" if self.multi_res_loss_weight > 0 else "off"
        self.multi_res_mode = str(train_cfg.get("multi_res_mode", default_multi_res_mode)).lower()
        raw_band_boundaries = train_cfg.get("multi_res_band_boundaries", train_cfg.get("multi_res_horizons", []))
        self.multi_res_band_boundaries = self._resolve_multi_res_boundaries(raw_band_boundaries, self.pred_len)
        self.multi_res_band_slices = self._build_multi_res_band_slices(self.multi_res_band_boundaries)
        self.multi_res_weight_mode = str(
            train_cfg.get(
                "multi_res_weight_mode",
                "adaptive" if self.multi_res_mode == "dynamic_band" else "off",
            )
        ).lower()
        self.multi_res_weight_focus = str(train_cfg.get("multi_res_weight_focus", "hard")).lower()
        self.multi_res_weight_beta = float(train_cfg.get("multi_res_weight_beta", 0.95))
        self.multi_res_weight_temp = float(train_cfg.get("multi_res_weight_temp", 1.0))
        self.multi_res_weight_strength = float(train_cfg.get("multi_res_weight_strength", 0.3))
        self.multi_res_weight_alpha = float(train_cfg.get("multi_res_weight_alpha", 0.7))
        self.multi_res_weight_floor = float(train_cfg.get("multi_res_weight_floor", 0.15))
        self.multi_res_weight_warmup_steps = int(train_cfg.get("multi_res_weight_warmup_steps", 400))
        self.multi_res_enabled = (
            (not self.noise_esti)
            and self.multi_res_loss_weight > 0
            and len(self.multi_res_band_slices) > 0
            and self.multi_res_mode != "off"
        )
        band_centers = []
        for start, end in self.multi_res_band_slices:
            band_centers.append((float(start) + float(end)) / 2.0 / max(float(self.pred_len), 1.0))
        if not band_centers:
            band_centers = [0.5]
        self.multi_res_band_labels = self._build_multi_res_band_labels(self.multi_res_band_slices)
        self.register_buffer(
            "multi_res_band_centers",
            torch.tensor(band_centers, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "multi_res_ema_losses",
            torch.zeros(len(band_centers), dtype=torch.float32),
            persistent=False,
        )
        self.multi_res_train_step = 0
        self.use_scale_router = bool(train_cfg.get("use_scale_router", False))
        self.scale_router_hidden_dim = int(train_cfg.get("scale_router_hidden_dim", 32))
        self.scale_router_dropout = float(train_cfg.get("scale_router_dropout", 0.1))
        self.scale_router_temp = float(train_cfg.get("scale_router_temp", max(self.multi_res_weight_temp, 1e-6)))
        self.scale_router_entropy_weight = float(train_cfg.get("scale_router_entropy_weight", 0.0))
        self.scale_router_use_trend_prior = bool(train_cfg.get("scale_router_use_trend_prior", True))
        self.scale_router_use_text_mask = bool(train_cfg.get("scale_router_use_text_mask", True))
        self.scale_router_teacher_weight = float(train_cfg.get("scale_router_teacher_weight", 0.0))
        self.scale_router_warmup_steps = int(train_cfg.get("scale_router_warmup_steps", self.multi_res_weight_warmup_steps))
        self.scale_router_enabled = (
            self.multi_res_enabled
            and len(self.multi_res_band_slices) > 1
            and self.use_scale_router
        )
        self.scale_router_train_step = 0
        self._last_scale_router_state = None
        self.use_router_guide = bool(config["diffusion"].get("use_router_guide", False))
        self.router_guide_alpha = float(config["diffusion"].get("router_guide_alpha", 0.0))
        self.router_guide_min_ratio = float(config["diffusion"].get("router_guide_min_ratio", 0.5))
        self.router_guide_max_ratio = float(config["diffusion"].get("router_guide_max_ratio", 1.5))
        self.router_guide_detach = bool(config["diffusion"].get("router_guide_detach", True))
        self.router_guide_enabled = self.cfg and self.scale_router_enabled and self.use_router_guide
        self._last_router_guide_state = None
        if self.scale_router_enabled:
            router_input_dim = 6
            if self.scale_router_use_trend_prior:
                router_input_dim += 3
            if self.scale_router_use_text_mask:
                router_input_dim += 1
            self.scale_router = ScaleRouter(
                input_dim=router_input_dim,
                output_dim=len(self.multi_res_band_slices),
                hidden_dim=self.scale_router_hidden_dim,
                dropout=self.scale_router_dropout,
            )
        else:
            self.scale_router = None

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1 
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
            
        if self.decomp:
            self.decomposition = CSDI_series_decomp(self.lookback_len, self.pred_len, kernel_size=25)

        if self.timestep_emb_cat:
            self.timestep_emb = nn.Sequential(nn.Linear(config["model"]["timestep_dim"], self.diff_channels//8), 
                                      nn.LayerNorm(self.diff_channels//8),
                                      nn.ReLU(),
                                      nn.Linear(self.diff_channels//8, self.diff_channels//4), 
                                      nn.LayerNorm(self.diff_channels//4),
                                      nn.ReLU())
        if self.timestep_branch:
            # Predict series directly from timestep features for TTF branch
            self.timestep_pred = nn.Sequential(
                nn.Conv1d(config["model"]["timestep_dim"], self.diff_channels, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(self.diff_channels, self.target_dim, kernel_size=1),
            )
        
        if self.relative_size_emb_cat:
            self.relative_size_emb = nn.Sequential(nn.Linear(self.lookback_len, self.lookback_len), 
                                                   nn.LayerNorm(self.lookback_len),
                                                   nn.ReLU(),
                                                   nn.Linear(self.lookback_len, self.diff_channels),
                                                   nn.LayerNorm(self.diff_channels),
                                                   nn.ReLU(),)

        if self.with_texts:
            self.text_encoder, self.tokenizer = get_llm(self.llm, config["model"]["llm_layers"])
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            if self.llm != 'bert':
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    pad_token = '[PAD]'
                    self.tokenizer.add_special_tokens({'pad_token': pad_token})
                    self.tokenizer.pad_token = pad_token

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        config_diff["decomp"] = self.decomp
        config_diff["lookback_len"] = self.lookback_len
        config_diff["pred_len"] = self.pred_len
        config_diff["with_timestep"] = True if self.timestep_emb_cat else False
        config_diff["context_dim"] = self.context_dim
        config_diff["with_texts"] = self.with_texts
        config_diff["time_weight"] = config["diffusion"]["time_weight"]
        config_diff["save_attn"] = config["model"]["save_attn"]

        if self.is_unconditional == True:
            input_dim = 1
        else:
            input_dim = 3 if self.self_condition else 2
        mode_num = 1

        if self.decomp:
            self.diffmodel_trend = diff_CSDI(config_diff, input_dim, mode_num=mode_num)
            self.diffmodel_sesonal = diff_CSDI(config_diff, input_dim, mode_num=mode_num)
        else:
            self.diffmodel = diff_CSDI(config_diff, input_dim, mode_num=mode_num)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else: 
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask


    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim) 
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K, emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1) 
        side_info = side_info.permute(0, 3, 2, 1) 

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1) 
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def _forward_diffmodel(self, total_input, side_info, diffusion_step, cfg_mask, timestep_emb=None, size_emb=None, context=None):
        if self.decomp:
            predicted_seasonal = self.diffmodel_sesonal(total_input[0], side_info, diffusion_step, cfg_mask, timestep_emb, size_emb)
            predicted_trend = self.diffmodel_trend(total_input[1], side_info, diffusion_step, cfg_mask, timestep_emb, size_emb)
            return predicted_seasonal + predicted_trend
        if self.save_attn:
            predicted, _ = self.diffmodel(total_input, side_info, diffusion_step, cfg_mask, timestep_emb, size_emb, context)
            return predicted
        return self.diffmodel(total_input, side_info, diffusion_step, cfg_mask, timestep_emb, size_emb, context)

    def _apply_timestep_branch(self, predicted, timesteps):
        if self.timestep_branch and timesteps is not None:
            predicted_from_timestep = self.timestep_pred(timesteps)
            return 0.9 * predicted + 0.1 * predicted_from_timestep
        return predicted

    def _build_self_condition(self, predicted, cond_mask):
        if predicted is None:
            return None
        self_cond = predicted.detach()
        if self.self_condition_target_only:
            self_cond = self_cond * (1.0 - cond_mask)
        return self_cond

    def calc_loss_valid(
        self,
        observed_data,
        cond_mask,
        observed_mask,
        side_info,
        is_train,
        timesteps=None,
        timestep_emb=None,
        size_emb=None,
        context=None,
        trend_prior=None,
        text_mask=None,
        scale_code=None,
    ):
        loss_sum = 0
        for t in range(self.num_steps): 
            loss = self.calc_loss(
                observed_data,
                cond_mask,
                observed_mask,
                side_info,
                is_train,
                set_t=t,
                timesteps=timesteps,
                timestep_emb=timestep_emb,
                size_emb=size_emb,
                context=context,
                trend_prior=trend_prior,
                text_mask=text_mask,
                scale_code=scale_code,
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self,
        observed_data,
        cond_mask,
        observed_mask,
        side_info,
        is_train,
        timesteps=None,
        timestep_emb=None,
        size_emb=None,
        context=None,
        trend_prior=None,
        text_mask=None,
        scale_code=None,
        set_t=-1,
    ):  
        
        B, K, L = observed_data.shape
        if not self.noise_esti:
            means = torch.sum(observed_data*cond_mask, dim=2, keepdim=True) / torch.sum(cond_mask, dim=2, keepdim=True)
            stdev = torch.sqrt(torch.sum((observed_data - means) ** 2 * cond_mask, dim=2, keepdim=True) / (torch.sum(cond_mask, dim=2, keepdim=True) - 1) + 1e-5)
            observed_data = (observed_data - means) / stdev

        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device) 
        current_alpha = self.alpha_torch[t]  
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        if self.cfg:
            cfg_mask = torch.bernoulli(torch.ones((B, )) - self.c_mask_prob).to(self.device) 
        else:
            cfg_mask = None

        self_cond = None
        use_self_condition = self.self_condition and (not self.is_unconditional) and (is_train != 1 or np.random.rand() < self.self_condition_prob)
        if use_self_condition:
            with torch.no_grad():
                preview_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask, self_cond=None)
                preview_pred = self._forward_diffmodel(preview_input, side_info, t, cfg_mask, timestep_emb, size_emb, context)
                preview_pred = self._apply_timestep_branch(preview_pred, timesteps)
                self_cond = self._build_self_condition(preview_pred, cond_mask)

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask, self_cond=self_cond) 
        predicted = self._forward_diffmodel(total_input, side_info, t, cfg_mask, timestep_emb, size_emb, context)
        predicted = self._apply_timestep_branch(predicted, timesteps)

        target_mask = observed_mask - cond_mask
        if self.noise_esti:
            residual = (noise - predicted) * target_mask 
        else:
            residual = (observed_data - predicted) * target_mask 
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        if self.multi_res_enabled and not self.noise_esti:
            aux_loss = self._calc_multi_res_loss(
                observed_data,
                predicted,
                target_mask,
                trend_prior=trend_prior,
                text_mask=text_mask,
                scale_code=scale_code,
                is_train=is_train,
            )
            loss = loss + self.multi_res_loss_weight * aux_loss
        return loss

    def _load_text_score_model(self, model_path):
        if (not self.use_text_score_gate) or (not model_path):
            return None
        try:
            with open(model_path, "r") as f:
                payload = json.load(f)
        except Exception:
            return None
        metrics = payload.get("metrics", payload)
        feature_columns = payload.get("feature_columns", [])
        weights = metrics.get("weights", {})
        feature_mean = metrics.get("feature_mean", {})
        feature_std = metrics.get("feature_std", {})
        target_mean = float(metrics.get("mean_target", 0.0))
        target_std = float(metrics.get("std_target", 1.0))
        if target_std <= 1e-8:
            target_std = 1.0
        if not feature_columns or "intercept" not in weights:
            return None
        return {
            "feature_columns": list(feature_columns),
            "weights": dict(weights),
            "feature_mean": dict(feature_mean),
            "feature_std": dict(feature_std),
            "target_mean": target_mean,
            "target_std": target_std,
        }

    def _compute_online_text_score(self, batch, text_mask, trend_prior, text_window_len, guide_w=None):
        base_score = text_mask.float()
        if not self.use_text_score_gate:
            return base_score
        explicit_score = batch.get("text_score")
        if explicit_score is not None:
            if torch.is_tensor(explicit_score):
                score = explicit_score.to(self.device).float().reshape(-1)
            else:
                score = torch.tensor(explicit_score, device=self.device, dtype=torch.float32).reshape(-1)
            return torch.clamp(score, 0.0, 1.0) * base_score
        if self.text_score_model is None:
            return base_score

        batch_size = int(text_mask.shape[0])
        raw_texts = batch.get("raw_text", batch.get("texts", ["NA"] * batch_size))
        full_texts = batch.get("texts", ["NA"] * batch_size)
        retrieved_texts = batch.get("retrieved_text", [""] * batch_size)
        cot_texts = batch.get("cot_text", [""] * batch_size)
        if isinstance(raw_texts, tuple):
            raw_texts = list(raw_texts)
        if isinstance(full_texts, tuple):
            full_texts = list(full_texts)
        if isinstance(retrieved_texts, tuple):
            retrieved_texts = list(retrieved_texts)
        if isinstance(cot_texts, tuple):
            cot_texts = list(cot_texts)
        if not isinstance(raw_texts, list):
            raw_texts = [raw_texts] * batch_size
        if not isinstance(full_texts, list):
            full_texts = [full_texts] * batch_size
        if not isinstance(retrieved_texts, list):
            retrieved_texts = [retrieved_texts] * batch_size
        if not isinstance(cot_texts, list):
            cot_texts = [cot_texts] * batch_size

        guide_value = 0.0 if guide_w is None else float(guide_w)
        feature_tensors = {
            "text_mark": text_mask.float().reshape(-1),
            "text_window_len": text_window_len.float().reshape(-1),
            "scale_code": batch.get("scale_code").to(self.device).float().reshape(-1) if torch.is_tensor(batch.get("scale_code")) else torch.zeros(batch_size, device=self.device),
            "raw_text_len": torch.tensor([len(str(item).split()) for item in raw_texts], device=self.device, dtype=torch.float32),
            "full_text_len": torch.tensor([len(str(item).split()) for item in full_texts], device=self.device, dtype=torch.float32),
            "retrieved_text_len": torch.tensor([len(str(item).split()) for item in retrieved_texts], device=self.device, dtype=torch.float32),
            "cot_text_len": torch.tensor([len(str(item).split()) for item in cot_texts], device=self.device, dtype=torch.float32),
            "trend_direction": trend_prior[:, 0].float(),
            "trend_strength": trend_prior[:, 1].float(),
            "trend_volatility": trend_prior[:, 2].float(),
            "guide_w": torch.full((batch_size,), guide_value, device=self.device, dtype=torch.float32),
        }

        linear = torch.full((batch_size,), float(self.text_score_model["weights"].get("intercept", 0.0)), device=self.device, dtype=torch.float32)
        for feature_name in self.text_score_model["feature_columns"]:
            value = feature_tensors.get(feature_name)
            if value is None:
                continue
            mean = float(self.text_score_model["feature_mean"].get(feature_name, 0.0))
            std = float(self.text_score_model["feature_std"].get(feature_name, 1.0))
            if abs(std) < 1e-8:
                std = 1.0
            weight = float(self.text_score_model["weights"].get(feature_name, 0.0))
            linear = linear + weight * ((value - mean) / std)
        target_mean = float(self.text_score_model["target_mean"])
        target_std = float(self.text_score_model["target_std"])
        score = torch.sigmoid((linear - target_mean) / max(target_std, 1e-6))
        return score * base_score

    def _apply_text_score_gate(self, text_score):
        if text_score is None:
            return None
        base = text_score.float()
        if not self.use_text_score_gate:
            return base
        floor = min(max(self.text_score_gate_floor, 0.0), 1.0)
        strength = min(max(self.text_score_gate_strength, 0.0), 1.0)
        gated = floor + (1.0 - floor) * base
        return ((1.0 - strength) + strength * gated) * (base > 0).float()

    def _resolve_multi_res_boundaries(self, raw_boundaries, pred_len):
        if raw_boundaries is None:
            return []
        if isinstance(raw_boundaries, int):
            raw_boundaries = [raw_boundaries]
        boundaries = []
        for item in raw_boundaries:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            value = min(value, int(pred_len))
            if value > 0:
                boundaries.append(value)
        boundaries = sorted(set(boundaries))
        if pred_len > 0 and (not boundaries or boundaries[-1] != int(pred_len)):
            boundaries.append(int(pred_len))
        return boundaries

    def _build_multi_res_band_slices(self, boundaries):
        band_slices = []
        left = 0
        for right in boundaries:
            right = int(right)
            if right <= left:
                continue
            band_slices.append((left, right))
            left = right
        return band_slices

    def _build_multi_res_band_labels(self, band_slices):
        labels = []
        for start, end in band_slices:
            left = int(start) + 1
            right = int(end)
            if left == right:
                labels.append(f"{left}")
            else:
                labels.append(f"{left}-{right}")
        return labels

    def _compute_router_scale_score(self, router_weights):
        centers = self.multi_res_band_centers.view(1, -1).to(router_weights.device)
        return (router_weights * centers).sum(dim=1)

    def get_multi_res_band_info(self):
        return list(zip(self.multi_res_band_labels, self.multi_res_band_slices))

    def _multi_res_pointwise_loss(self, residual):
        if self.multi_res_use_huber:
            delta = max(self.multi_res_huber_delta, 1e-6)
            abs_res = residual.abs()
            return torch.where(
                abs_res <= delta,
                0.5 * residual ** 2,
                delta * abs_res - 0.5 * (delta ** 2),
            )
        return residual ** 2

    def _compute_multi_res_band_losses(self, observed_data, predicted, target_mask):
        if len(self.multi_res_band_slices) == 0:
            zero = torch.zeros((), device=observed_data.device)
            return zero.unsqueeze(0), zero.unsqueeze(0).unsqueeze(0), zero.unsqueeze(0).unsqueeze(0)
        residual = observed_data - predicted
        band_means = []
        band_sample_losses = []
        band_valid_mask = []
        base_index = int(self.lookback_len)
        for start, end in self.multi_res_band_slices:
            abs_start = base_index + int(start)
            abs_end = base_index + int(end)
            band_mask = target_mask[:, :, abs_start:abs_end]
            band_residual = residual[:, :, abs_start:abs_end]
            pointwise = self._multi_res_pointwise_loss(band_residual) * band_mask
            per_sample_num = band_mask.sum(dim=(1, 2))
            per_sample_loss = pointwise.sum(dim=(1, 2)) / per_sample_num.clamp(min=1.0)
            valid = per_sample_num > 0
            valid_float = valid.float()
            band_mean = (per_sample_loss * valid_float).sum() / valid_float.sum().clamp(min=1.0)
            band_means.append(band_mean)
            band_sample_losses.append(per_sample_loss)
            band_valid_mask.append(valid_float)
        return (
            torch.stack(band_means, dim=0),
            torch.stack(band_sample_losses, dim=1),
            torch.stack(band_valid_mask, dim=1),
        )

    def _normalize_multi_res_feature(self, values):
        denom = values.detach().mean().clamp(min=1e-6)
        return (values / denom).clamp(min=0.0, max=4.0)

    def _extract_scale_router_features(self, observed_data, trend_prior=None, text_mask=None):
        history = observed_data[:, :, : self.lookback_len]
        batch_size = history.shape[0]
        if history.shape[-1] <= 1:
            features = history.new_zeros((batch_size, 6))
        else:
            signed_slope = (history[:, :, -1] - history[:, :, 0]).mean(dim=1)
            abs_slope = signed_slope.abs()
            volatility = history.std(dim=2, unbiased=False).mean(dim=1)
            diffs = history[:, :, 1:] - history[:, :, :-1]
            diff_std = diffs.std(dim=2, unbiased=False).mean(dim=1) if diffs.shape[-1] > 0 else torch.zeros_like(volatility)
            if diffs.shape[-1] > 1:
                accel = (diffs[:, :, 1:] - diffs[:, :, :-1]).abs().mean(dim=(1, 2))
            else:
                accel = torch.zeros_like(volatility)
            mean_abs = history.abs().mean(dim=(1, 2))
            features = torch.stack(
                [
                    signed_slope / (volatility + 1e-6),
                    abs_slope / (volatility + 1e-6),
                    volatility / (mean_abs + 1e-6),
                    diff_std / (volatility + 1e-6),
                    accel / (diff_std + 1e-6),
                    torch.log1p(mean_abs),
                ],
                dim=1,
            )
            features = torch.nan_to_num(features, nan=0.0, posinf=4.0, neginf=-4.0)

        extra_features = [features]
        if self.scale_router_use_trend_prior:
            if trend_prior is None:
                trend_values = history.new_zeros((batch_size, 3))
            else:
                trend_values = trend_prior[:, :3]
            extra_features.append(trend_values)
        if self.scale_router_use_text_mask:
            if text_mask is None:
                text_values = history.new_zeros((batch_size, 1))
            else:
                text_values = text_mask.float().reshape(-1, 1)
            extra_features.append(text_values)
        return torch.cat(extra_features, dim=1)

    def _build_scale_router_teacher(self, scale_code, device):
        if scale_code is None or len(self.multi_res_band_slices) <= 1:
            return None, None
        anchors = scale_code.float().clamp(min=0.0, max=2.0) / 2.0
        centers = self.multi_res_band_centers.view(1, -1).to(device)
        distances = torch.abs(centers - anchors.unsqueeze(1))
        target_index = distances.argmin(dim=1)
        teacher_probs = torch.softmax(-distances * max(len(self.multi_res_band_slices), 1), dim=1)
        return target_index, teacher_probs

    def _compute_scale_router_weights(self, observed_data, trend_prior=None, text_mask=None, scale_code=None, is_train=1):
        if not self.scale_router_enabled or self.scale_router is None:
            return None

        features = self._extract_scale_router_features(
            observed_data,
            trend_prior=trend_prior,
            text_mask=text_mask,
        )
        logits = self.scale_router(features) / max(self.scale_router_temp, 1e-6)
        weights = torch.softmax(logits, dim=1)
        entropy = -(weights * torch.log(weights.clamp(min=1e-6))).sum(dim=1)

        target_index, teacher_probs = self._build_scale_router_teacher(scale_code, observed_data.device)
        teacher_loss = weights.new_zeros(())
        if is_train == 1:
            self.scale_router_train_step += 1
            if (
                self.scale_router_teacher_weight > 0
                and teacher_probs is not None
                and self.scale_router_warmup_steps > 0
            ):
                warmup_ratio = max(
                    0.0,
                    1.0 - float(self.scale_router_train_step) / max(float(self.scale_router_warmup_steps), 1.0),
                )
                if warmup_ratio > 0.0:
                    teacher_loss = (
                        self.scale_router_teacher_weight
                        * warmup_ratio
                        * F.kl_div(
                            torch.log(weights.clamp(min=1e-6)),
                            teacher_probs.detach(),
                            reduction="batchmean",
                        )
                    )

        entropy_reg = weights.new_zeros(())
        if self.scale_router_entropy_weight != 0.0:
            entropy_reg = -self.scale_router_entropy_weight * entropy.mean()

        self._last_scale_router_state = {
            "weights": weights.detach(),
            "entropy": entropy.detach(),
            "target_index": None if target_index is None else target_index.detach(),
        }
        return {
            "weights": weights,
            "reg_loss": teacher_loss + entropy_reg,
            "entropy": entropy,
            "target_index": target_index,
        }

    def inspect_scale_router(self, observed_data, trend_prior=None, text_mask=None, scale_code=None, text_window_len=None):
        if not self.scale_router_enabled:
            return None
        with torch.no_grad():
            router_state = self._compute_scale_router_weights(
                observed_data,
                trend_prior=trend_prior,
                text_mask=text_mask,
                scale_code=scale_code,
                is_train=0,
            )
            if router_state is None:
                return None
            diagnostics = {
                "weights": router_state["weights"].detach().cpu(),
                "argmax": router_state["weights"].argmax(dim=1).detach().cpu(),
                "entropy": router_state["entropy"].detach().cpu(),
            }
            if router_state["target_index"] is not None:
                diagnostics["target_index"] = router_state["target_index"].detach().cpu()
            if scale_code is not None:
                diagnostics["scale_code"] = scale_code.detach().cpu()
            if text_window_len is not None:
                diagnostics["text_window_len"] = text_window_len.detach().cpu()
            return diagnostics

    def _compute_router_guidance(self, observed_data, guide_w, trend_prior=None, text_mask=None):
        self._last_router_guide_state = None
        if (not self.router_guide_enabled) or guide_w == 0:
            return None
        router_state = self._compute_scale_router_weights(
            observed_data,
            trend_prior=trend_prior,
            text_mask=text_mask,
            scale_code=None,
            is_train=0,
        )
        if router_state is None:
            return None
        router_weights = router_state["weights"]
        if self.router_guide_detach:
            router_weights = router_weights.detach()
        centers = self.multi_res_band_centers.view(1, -1).to(observed_data.device)
        scale_score = (router_weights * centers).sum(dim=1)
        guide_ratio = 1.0 + self.router_guide_alpha * (scale_score - 0.5)
        min_ratio = min(self.router_guide_min_ratio, self.router_guide_max_ratio)
        max_ratio = max(self.router_guide_min_ratio, self.router_guide_max_ratio)
        guide_ratio = guide_ratio.clamp(min=min_ratio, max=max_ratio)
        sample_guide = guide_ratio * float(guide_w)
        self._last_router_guide_state = {
            "sample_guide": sample_guide.detach(),
            "guide_ratio": guide_ratio.detach(),
            "scale_score": scale_score.detach(),
            "weights": router_weights.detach(),
        }
        return sample_guide

    def _compute_multi_res_sample_weights(self, observed_data, trend_prior=None, text_mask=None):
        batch_size = observed_data.shape[0]
        num_bands = len(self.multi_res_band_slices)
        if num_bands == 0:
            return torch.ones((batch_size, 1), device=observed_data.device)

        history = observed_data[:, :, : self.lookback_len]
        if history.shape[-1] <= 1:
            return torch.full(
                (batch_size, num_bands),
                1.0 / max(num_bands, 1),
                device=observed_data.device,
            )

        slope = (history[:, :, -1] - history[:, :, 0]).abs().mean(dim=1)
        volatility = history.std(dim=2, unbiased=False).mean(dim=1)
        diffs = history[:, :, 1:] - history[:, :, :-1]
        if diffs.shape[-1] > 1:
            accel = (diffs[:, :, 1:] - diffs[:, :, :-1]).abs().mean(dim=(1, 2))
        else:
            accel = torch.zeros_like(slope)

        if trend_prior is None:
            trend_strength = torch.ones_like(slope)
            trend_volatility = torch.zeros_like(slope)
        else:
            trend_strength = trend_prior[:, 1].clamp(min=0.0)
            trend_volatility = trend_prior[:, 2].clamp(min=0.0)
        if text_mask is None:
            text_signal = torch.zeros_like(slope)
        else:
            text_signal = text_mask.float().reshape(-1)
        short_signal = (
            self._normalize_multi_res_feature(volatility)
            + 0.5 * self._normalize_multi_res_feature(accel)
            + 0.25 * self._normalize_multi_res_feature(trend_volatility)
        )
        long_signal = (
            self._normalize_multi_res_feature(slope)
            + 0.25 * self._normalize_multi_res_feature(trend_strength)
            + 0.1 * text_signal
        )
        scale_pref = long_signal / (short_signal + long_signal + 1e-6)
        centers = self.multi_res_band_centers.view(1, -1).to(observed_data.device)
        dist = torch.abs(centers - scale_pref.unsqueeze(1))
        temp = max(self.multi_res_weight_temp, 1e-6)
        scores = -dist * max(num_bands, 1) / temp
        return torch.softmax(scores, dim=1)

    def _get_multi_res_global_weights(self, band_losses, is_train):
        num_bands = band_losses.shape[0]
        if num_bands == 0:
            return band_losses.new_ones((1,))
        uniform = band_losses.new_full((num_bands,), 1.0 / max(num_bands, 1))
        if self.multi_res_weight_mode != "adaptive":
            return uniform

        if is_train == 1:
            with torch.no_grad():
                if self.multi_res_train_step == 0:
                    self.multi_res_ema_losses.copy_(band_losses.detach())
                else:
                    beta = min(max(self.multi_res_weight_beta, 0.0), 0.9999)
                    self.multi_res_ema_losses.mul_(beta).add_((1.0 - beta) * band_losses.detach())
                self.multi_res_train_step += 1

        if self.multi_res_train_step < self.multi_res_weight_warmup_steps:
            return uniform

        logits = self.multi_res_ema_losses.clone().to(band_losses.device)
        if self.multi_res_weight_focus == "easy":
            logits = -logits
        logits = logits / max(self.multi_res_weight_temp, 1e-6)
        return torch.softmax(logits, dim=0)

    def _calc_multi_res_loss(self, observed_data, predicted, target_mask, trend_prior=None, text_mask=None, scale_code=None, is_train=1):
        band_losses, band_sample_losses, band_valid = self._compute_multi_res_band_losses(
            observed_data,
            predicted,
            target_mask,
        )
        if band_losses.numel() == 0:
            return torch.zeros((), device=observed_data.device)

        base_loss = band_losses.mean()
        router_state = self._compute_scale_router_weights(
            observed_data,
            trend_prior=trend_prior,
            text_mask=text_mask,
            scale_code=scale_code,
            is_train=is_train,
        )
        if self.multi_res_weight_mode != "adaptive" and router_state is None:
            return base_loss

        global_weights = self._get_multi_res_global_weights(band_losses, is_train=is_train)
        if router_state is None and self.multi_res_train_step < self.multi_res_weight_warmup_steps:
            return base_loss

        if router_state is not None:
            sample_weights = router_state["weights"]
        else:
            sample_weights = self._compute_multi_res_sample_weights(
                observed_data,
                trend_prior=trend_prior,
                text_mask=text_mask,
            )
        alpha = min(max(self.multi_res_weight_alpha, 0.0), 1.0)
        strength = min(max(self.multi_res_weight_strength, 0.0), 1.0)
        floor = min(max(self.multi_res_weight_floor, 0.0), 1.0)
        uniform = band_sample_losses.new_full(sample_weights.shape, 1.0 / sample_weights.shape[1])
        mixed_dynamic = alpha * sample_weights + (1.0 - alpha) * global_weights.unsqueeze(0)
        mixed_weights = (1.0 - strength) * uniform + strength * mixed_dynamic
        final_weights = (1.0 - floor) * mixed_weights + floor * uniform
        final_weights = final_weights * band_valid
        final_weights = final_weights / final_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weighted_loss = (final_weights * band_sample_losses).sum(dim=1)
        valid_samples = (band_valid.sum(dim=1) > 0).float()
        aux_loss = (weighted_loss * valid_samples).sum() / valid_samples.sum().clamp(min=1.0)
        if router_state is not None:
            aux_loss = aux_loss + router_state["reg_loss"]
        return aux_loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask, self_cond=None, mask_noisy_target=False):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  
        else:
            cond_obs = cond_mask * observed_data
            target_source = (1 - cond_mask) * noisy_data if mask_noisy_target else noisy_data
            noisy_target = target_source.unsqueeze(1) 
            if self.self_condition:
                if self_cond is None:
                    self_cond = torch.zeros_like(observed_data)
                elif self.self_condition_target_only:
                    self_cond = self_cond * (1.0 - cond_mask)
                self_cond = self_cond.unsqueeze(1)
            if self.decomp:
                res, moving_mean = self.decomposition(cond_obs) 
                res, moving_mean = res.unsqueeze(1), moving_mean.unsqueeze(1) 
                res_parts = [res, noisy_target]
                moving_mean_parts = [moving_mean, noisy_target]
                if self.self_condition:
                    res_parts.append(self_cond)
                    moving_mean_parts.append(self_cond)
                res_input = torch.cat(res_parts, dim=1)  
                moving_mean_input = torch.cat(moving_mean_parts, dim=1) 
                total_input = [res_input, moving_mean_input]
            else:
                cond_obs = cond_obs.unsqueeze(1) 
                input_parts = [cond_obs, noisy_target]
                if self.self_condition:
                    input_parts.append(self_cond)
                total_input = torch.cat(input_parts, dim=1) 

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=None, timestep_emb=None, size_emb=None, context=None, trend_prior=None, text_mask=None):
        B, K, L = observed_data.shape
        if self.ddim:
            if self.sample_method == 'linear':
                a = self.num_steps // self.sample_steps
                time_steps = np.asarray(list(range(0, self.num_steps, a)))
            elif self.sample_method == "quad":
                time_steps = (np.linspace(0, np.sqrt(self.num_steps * 0.8), self.sample_steps) ** 2).astype(int)
            else:
                raise NotImplementedError(f"sampling method {self.sample_method} is not implemented!")
            time_steps = time_steps + 1
            time_steps_prev = np.concatenate([[0], time_steps[:-1]])
        else:
            self.sample_steps = self.num_steps
        if not self.noise_esti:
            means = torch.sum(observed_data*cond_mask, dim=2, keepdim=True) / torch.sum(cond_mask, dim=2, keepdim=True)
            stdev = torch.sqrt(torch.sum((observed_data - means) ** 2 * cond_mask, dim=2, keepdim=True) / (torch.sum(cond_mask, dim=2, keepdim=True) - 1) + 1e-5)
            observed_data = (observed_data - means) / stdev
        sample_guide = self._compute_router_guidance(
            observed_data,
            guide_w,
            trend_prior=trend_prior,
            text_mask=text_mask,
        )
        
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        if self.cfg:
            side_info = side_info.repeat(2, 1, 1, 1)
            if timestep_emb is not None:
                timestep_emb = timestep_emb.repeat(2, 1, 1, 1)
            if context is not None:
                context = context.repeat(2, 1, 1)
            cfg_mask = torch.zeros((2*B, )).to(self.device) 
            cfg_mask[:B] = 1.
        else:
            cfg_mask = None

        for i in range(n_samples):
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)
            self_cond = None
            for t in range(self.sample_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1) 
                    predicted = self._forward_diffmodel(
                        diff_input,
                        side_info,
                        torch.tensor([t]).to(self.device),
                        cfg_mask,
                        timestep_emb,
                        size_emb,
                        context,
                    )
                else:
                    diff_input = self.set_input_to_diffmodel(
                        current_sample,
                        observed_data,
                        cond_mask,
                        self_cond=self_cond,
                        mask_noisy_target=True,
                    )
                    if self.cfg:
                        if self.decomp:
                            diff_input = [item.repeat(2, 1, 1, 1) for item in diff_input]
                        else:
                            diff_input = diff_input.repeat(2, 1, 1, 1)
                    predicted = self._forward_diffmodel(
                        diff_input,
                        side_info,
                        torch.tensor([t]).to(self.device),
                        cfg_mask,
                        timestep_emb,
                        size_emb,
                        context,
                    )
                if self.cfg:
                    predicted_cond, predicted_uncond = predicted[:B], predicted[B:]
                    if self.trend_cfg:
                        if self.trend_cfg_random:
                            trend_prior = self.sample_random_trend_prior(B, observed_data.device)
                        if trend_prior is not None:
                            step_ratio = self.get_trend_step_ratio(t, time_steps if self.ddim else None)
                            guide_value = sample_guide if sample_guide is not None else guide_w
                            trend_weight = self.get_trend_guidance_weight(trend_prior, step_ratio, guide_value, text_mask)
                            predicted = predicted_uncond + trend_weight[:, None, None] * (predicted_cond - predicted_uncond)
                        else:
                            if sample_guide is not None:
                                predicted = predicted_uncond + sample_guide[:, None, None] * (predicted_cond - predicted_uncond)
                            else:
                                predicted = predicted_uncond + guide_w * (predicted_cond - predicted_uncond)
                    else:
                        if sample_guide is not None:
                            predicted = predicted_uncond + sample_guide[:, None, None] * (predicted_cond - predicted_uncond)
                        else:
                            predicted = predicted_uncond + guide_w * (predicted_cond - predicted_uncond)

                if self.self_condition and (not self.is_unconditional):
                    self_cond = self._build_self_condition(predicted, cond_mask)

                if self.noise_esti:
                    # noise prediction
                    if not self.ddim:
                        coeff1 = 1 / self.alpha_hat[t] ** 0.5
                        coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                        current_sample = coeff1 * (current_sample - coeff2 * predicted) # (B, K, L)
                        if t > 0:
                            noise = torch.randn_like(current_sample)
                            sigma = (
                                (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                            current_sample += sigma * noise
                    else:
                        tau, tau_prev = time_steps[t], time_steps_prev[t]
                        current_sample = (
                            torch.sqrt(self.alpha[tau_prev] / self.alpha[tau]) * current_sample +
                            (torch.sqrt(1 - self.alpha[tau_prev]) - torch.sqrt(
                                (self.alpha[tau_prev] * (1 - self.alpha[tau])) / self.alpha[tau])) * predicted
                        )
                else:
                    if not self.ddim:
                        if t > 1:
                            # data prediction
                            coeff1 = (self.alpha_hat[t] ** 0.5 * (1 - self.alpha[t-1])) / (1 - self.alpha[t])
                            coeff2 = (self.alpha[t-1] ** 0.5 * self.beta[t]) / (1 - self.alpha[t])
                            current_sample = coeff1 * current_sample + coeff2 * predicted # (B, K, L)
                            
                            if t > 2:
                                noise = torch.randn_like(current_sample)
                                sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                                ) ** 0.5
                                current_sample += sigma * noise
                    else:
                        tau, tau_prev = time_steps[t], time_steps_prev[t]
                        aaa_ = (1-self.alpha[tau_prev])/(1-self.alpha[tau]) ** 0.5
                        current_sample = (
                            aaa_ * current_sample +
                            ((self.alpha[tau_prev])**0.5 - (self.alpha[tau])**0.5 * aaa_) * predicted
                        )

            imputed_samples[:, i] = current_sample.detach()
            if self.timestep_branch and timesteps is not None:
                predicted_from_timestep = self.timestep_pred(timesteps)
                imputed_samples[:, i] = 0.9 * imputed_samples[:, i] + 0.1 * predicted_from_timestep.detach()
            if not self.noise_esti:
                imputed_samples[:, i] = imputed_samples[:, i] * stdev + means
        if self.save_attn:
            return imputed_samples, attn 
        else:
            return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)): 
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim, window_lens):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device, window_lens)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]
        

    def process_data(self, batch, guide_w=None):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        text_mask = batch["text_mark"].to(self.device).float()
        scale_code = batch.get("scale_code")
        if scale_code is None:
            scale_code = torch.full((observed_data.shape[0],), 1, device=self.device, dtype=torch.long)
        else:
            scale_code = scale_code.to(self.device).long()
        text_window_len = batch.get("text_window_len")
        if text_window_len is None:
            text_window_len = torch.full((observed_data.shape[0],), int(self.lookback_len), device=self.device, dtype=torch.long)
        else:
            text_window_len = text_window_len.to(self.device).long()
        trend_prior = batch.get("trend_prior")
        if trend_prior is None:
            trend_prior = torch.zeros((observed_data.shape[0], 3), device=self.device)
        else:
            trend_prior = trend_prior.to(self.device).float()
        if self.timestep_emb_cat or self.timestep_branch:
            timesteps = batch["timesteps"].to(self.device).float()
            timesteps = timesteps.permute(0, 2, 1)
        else:
            timesteps = None
        texts = list(batch["texts"]) if self.with_texts else None
        text_score = self._compute_online_text_score(batch, text_mask, trend_prior, text_window_len, guide_w=guide_w)
        text_mask = self._apply_text_score_gate(text_score)

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data, 
            observed_mask,
            observed_tp, 
            gt_mask,
            for_pattern_mask, 
            cut_length,
            feature_id,
            timesteps, 
            texts,
            text_mask,
            trend_prior,
            scale_code,
            text_window_len,
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask
    
    def get_timestep_info(self, timesteps):
        timestep_emb = self.timestep_emb(timesteps.transpose(1, 2)).transpose(1, 2)
        timestep_emb = timestep_emb.unsqueeze(2).expand(-1, -1, self.target_dim, -1) 
        return timestep_emb
    
    def get_relative_size_info(self, observed_data):
        B, K, L = observed_data.shape

        size_emb = observed_data[:, :, :self.lookback_len].clone().unsqueeze(3).expand(-1, -1, -1, self.lookback_len) - \
            observed_data[:, :, :self.lookback_len].clone().unsqueeze(2).expand(-1, -1, self.lookback_len, -1) 
        size_emb = self.relative_size_emb(size_emb)
        size_emb = size_emb.permute(0, 3, 1, 2)
        size_emb = torch.cat([size_emb, torch.zeros((B, self.diff_channels, K, self.pred_len)).to(observed_data.device)], dim=-1) 
        return size_emb

    def get_trend_step_ratio(self, step_index, time_steps=None):
        if self.ddim and time_steps is not None:
            current_step = float(time_steps[step_index])
        else:
            current_step = float(step_index)
        denom = max(self.num_steps - 1, 1)
        ratio = 1.0 - current_step / denom
        ratio = ratio ** self.trend_cfg_power
        floor = max(self.trend_time_floor, 0.0)
        if floor > 0.0:
            ratio = floor + (1.0 - floor) * ratio
        return ratio

    def get_trend_guidance_weight(self, trend_prior, step_ratio, guide_w, text_mask=None):
        strength = trend_prior[:, 1].clamp(min=0.0)
        strength = 1.0 + self.trend_strength_scale * (strength - 1.0)
        strength = strength.clamp(min=0.0)
        volatility = trend_prior[:, 2].clamp(min=0.0) * self.trend_volatility_scale
        vol_penalty = 1.0 / (1.0 + volatility)
        weight = guide_w * step_ratio * strength * vol_penalty
        if text_mask is not None:
            weight = weight * text_mask
        return weight

    def sample_random_trend_prior(self, batch_size, device):
        direction = torch.randint(0, 3, (batch_size,), device=device).float() - 1.0
        strength_choices = torch.tensor([0.5, 1.0, 1.5], device=device)
        volatility_choices = torch.tensor([0.0, 0.5, 1.0], device=device)
        strength = strength_choices[torch.randint(0, 3, (batch_size,), device=device)]
        volatility = volatility_choices[torch.randint(0, 3, (batch_size,), device=device)]
        return torch.stack([direction, strength, volatility], dim=1)
    
    def get_text_info(self, text, text_mask):
        token_input = self.tokenizer(text,
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors='pt',
                                     ).to(self.device)
        context = self.text_encoder(**token_input).last_hidden_state
        context = context * text_mask.unsqueeze(1).unsqueeze(1)
        context = context.permute(0, 2, 1) 
        if self.save_token:
            tokens_str = self.tokenizer.batch_decode(token_input['input_ids'])
            return context, tokens_str
        else:
            return context

    def get_side_info(self, observed_tp, cond_mask, feature_id=None, timesteps=None, texts=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim) 
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1) 

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            ) 
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else: 
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1) 

        side_info = torch.cat([time_embed, feature_embed], dim=-1) 
        side_info = side_info.permute(0, 3, 2, 1) 

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1) 
            side_info = torch.cat([side_info, side_mask], dim=1) 
    

        return side_info

    def forward(self, batch, is_train=1):
        data = self.process_data(batch, guide_w=None)
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
            timesteps,
            texts,
            text_mask,
            trend_prior,
            scale_code,
            _,
        ) = data
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id, timesteps, texts)

        if self.timestep_emb_cat:
            timestep_emb = self.get_timestep_info(timesteps)
        else:
            timestep_emb = None

        if self.relative_size_emb_cat:
            size_emb = self.get_relative_size_info(observed_data)
        else:
            size_emb = None

        if self.with_texts:
            if self.save_token:
                context, _ = self.get_text_info(texts, text_mask)
            else:
                context = self.get_text_info(texts, text_mask)
        else:
            context = None

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(
            observed_data,
            cond_mask,
            observed_mask,
            side_info,
            is_train,
            timesteps=timesteps,
            timestep_emb=timestep_emb,
            size_emb=size_emb,
            context=context,
            trend_prior=trend_prior,
            text_mask=text_mask,
            scale_code=scale_code,
        )

    def evaluate(self, batch, n_samples, guide_w):
        data = self.process_data(batch, guide_w=guide_w)
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
            timesteps,
            texts,
            text_mask,
            trend_prior,
            _,
            _,
        ) = data

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask, timesteps=timesteps, texts=texts)

            if self.timestep_emb_cat:
                timestep_emb = self.get_timestep_info(timesteps)
            else:
                timestep_emb = None

            if self.relative_size_emb_cat:
                size_emb = self.get_relative_size_info(observed_data)
            else:
                size_emb = None

            if self.with_texts:
                if self.save_token:
                    context, tokens = self.get_text_info(texts, text_mask)
                else:
                    context = self.get_text_info(texts, text_mask)
            else:
                context = None
            text_mask_f = text_mask.float() if text_mask is not None else None
            if self.save_attn:
                samples, attn = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=context, trend_prior=trend_prior, text_mask=text_mask_f)
            else:
                samples = self.impute(observed_data, cond_mask, side_info, n_samples, guide_w, timesteps=timesteps, timestep_emb=timestep_emb, size_emb=size_emb, context=context, trend_prior=trend_prior, text_mask=text_mask_f)

        if self.save_attn:
            if self.save_token:
                return samples, observed_data, target_mask, observed_mask, observed_tp, attn, tokens
            else:
                return samples, observed_data, target_mask, observed_mask, observed_tp, attn
        else:
            return samples, observed_data, target_mask, observed_mask, observed_tp

    def get_scale_router_diagnostics(self, batch, guide_w=None):
        if not self.scale_router_enabled:
            return None
        data = self.process_data(batch, guide_w=guide_w)
        (
            observed_data,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            text_mask,
            trend_prior,
            scale_code,
            text_window_len,
        ) = data
        diagnostics = self.inspect_scale_router(
            observed_data,
            trend_prior=trend_prior,
            text_mask=text_mask.float(),
            scale_code=scale_code,
            text_window_len=text_window_len,
        )
        if diagnostics is None or guide_w is None:
            return diagnostics
        if not self.router_guide_enabled:
            return diagnostics
        history = observed_data[:, :, : self.lookback_len]
        means = torch.sum(history, dim=2, keepdim=True) / max(history.shape[2], 1)
        centered = history - means
        history_std = torch.sqrt(torch.mean(centered ** 2, dim=2, keepdim=True) + 1e-5)
        normalized_observed = observed_data.clone()
        normalized_observed[:, :, : self.lookback_len] = centered / history_std
        router_guide = self._compute_router_guidance(
            normalized_observed,
            guide_w,
            trend_prior=trend_prior,
            text_mask=text_mask.float(),
        )
        if router_guide is not None and self._last_router_guide_state is not None:
            diagnostics["sample_guide_w"] = self._last_router_guide_state["sample_guide"].cpu()
            diagnostics["guide_ratio"] = self._last_router_guide_state["guide_ratio"].cpu()
            diagnostics["scale_score"] = self._last_router_guide_state["scale_score"].cpu()
        return diagnostics


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
