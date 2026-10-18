import numpy as np

_DEFAULT_METRICS = ("ssim", "radar_metrics")
_SUPPORTED_METRICS = frozenset((*_DEFAULT_METRICS, "csi", "bias", "hss"))


def _validate_metrics(metrics):
    if isinstance(metrics, str):
        metrics = (metrics,)
    metrics = tuple(metrics)
    if not metrics:
        raise ValueError("At least one evaluation metric must be selected.")
    unsupported = set(metrics) - _SUPPORTED_METRICS
    if unsupported:
        raise ValueError(f"Unsupported metrics: {', '.join(sorted(unsupported))}.")
    return metrics


def _validate_arrays(pred, true, check_finite=True):
    pred, true = np.asarray(pred), np.asarray(true)
    if pred.shape != true.shape or pred.ndim != 5 or pred.shape[2] != 1:
        raise ValueError("Prediction and target must share shape [B,T,1,H,W].")
    if any(size == 0 for size in pred.shape):
        raise ValueError("The evaluation arrays must not be empty.")
    if not all(np.issubdtype(x.dtype, np.number) and not np.iscomplexobj(x)
               for x in (pred, true)):
        raise ValueError("Prediction and target must contain real numeric values.")
    if check_finite and (not np.all(np.isfinite(pred)) or not np.all(np.isfinite(true))):
        raise ValueError("Prediction or target contains NaN/Inf.")
    return pred, true


def _format_log(result):
    return ", ".join(f"{key}: {value:.4f}" for key, value in result.items())


def _apply_spatial_tolerance(binary_map, pool_size):
    """Spatial max-pooling for neighborhood-based categorical evaluation (CSI-pool4 / CSI-pool16)."""
    if pool_size <= 1:
        return binary_map
    x = np.asarray(binary_map)
    if x.ndim == 5:
        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width)
    elif x.ndim == 4:
        batch, steps, height, width = x.shape
        x = x.reshape(batch * steps, 1, height, width)
    elif x.ndim == 3:
        batch, height, width = x.shape
        x = x.reshape(batch, 1, height, width)
    elif x.ndim == 2:
        x = x[None, None]
    else:
        raise ValueError("Expected 2D-5D spatial maps.")
    height, width = x.shape[-2:]
    pooled_h, pooled_w = height // pool_size, width // pool_size
    if not pooled_h or not pooled_w:
        raise ValueError(f"Map {height}x{width} is smaller than pool size {pool_size}.")
    x = x[..., :pooled_h * pool_size, :pooled_w * pool_size]
    return x.reshape(*x.shape[:-2], pooled_h, pool_size, pooled_w, pool_size).max(axis=(-3, -1))


def _threshold(x, y, t):
    t_map = np.greater_equal(x, t).astype(np.float32)
    p_map = np.greater_equal(y, t).astype(np.float32)
    return t_map, p_map


def calculate_confusion(pred, true, threshold):
    t_map, p_map = _threshold(pred, true, threshold)
    hits = np.sum(t_map * p_map)
    fas = np.sum(t_map * (1.0 - p_map))
    misses = np.sum((1.0 - t_map) * p_map)
    return hits, fas, misses


def BIAS(predicted_count, observed_count):
    if observed_count == 0:
        return float("nan")
    return float(predicted_count / observed_count)


def CSI(hits, fas, misses, eps=1e-6):
    return hits / (hits + misses + fas + eps)


def HSS(hits, fas, misses, cn, eps=1e-6):
    num = 2.0 * (hits * cn - misses * fas)
    den = (hits + misses) * (misses + cn) + (hits + fas) * (fas + cn)
    return num / (den + eps)


class RadarMetricAccumulator:
    """Streaming confusion matrix accumulator for dataset-wide categorical metrics (CSI, HSS, BIAS)."""

    _CONFIGS = {
        "cikm": (95.0, -10.0, (20, 30, 35, 40)),
        "meteonet": (70.0, 0.0, (12, 18, 24, 32)),
        "shanghai": (90.0, 0.0, (20, 30, 35, 40)),
        "sevir": (47.54, 33.44, (16, 74, 133, 160, 181, 219)),
        "default": (1.0, 0.0, (0.1, 0.25, 0.5)),
    }

    def __init__(self, dataset_name="default", pool_sizes=(1, 4, 16)):
        self.dataset_name = str(dataset_name)
        name = self.dataset_name.lower()
        scale, offset, thresholds = self._CONFIGS["default"]
        for key, values in self._CONFIGS.items():
            if key in name:
                scale, offset, thresholds = values
                break
        self.scale = scale
        self.offset = offset
        self.thresholds = thresholds
        if "meteonet" in name:
            self.max_val = 70.0
        elif "shanghai" in name:
            self.max_val = 90.0
        elif "cikm" in name:
            self.max_val = 85.0
        elif "sevir" in name:
            self.max_val = 255.0
        else:
            self.max_val = 1.0
        self.pool_sizes = tuple(pool_sizes)
        if 1 not in self.pool_sizes or any(pool not in (1, 4, 16) for pool in self.pool_sizes):
            raise ValueError("pool_sizes must include 1 and contain only 1, 4, or 16.")
        self.reset()

    def reset(self):
        self.counts = {pool: [[0, 0, 0, 0] for _ in self.thresholds] for pool in self.pool_sizes}
        self.sample_count = 0
        self.batch_count = 0
        self._frame_shape = None
        return self

    @staticmethod
    def _confusion(predicted, observed):
        tp = int(np.count_nonzero(predicted & observed))
        predicted_count = int(np.count_nonzero(predicted))
        observed_count = int(np.count_nonzero(observed))
        fp = predicted_count - tp
        fn = observed_count - tp
        tn = int(predicted.size) - tp - fp - fn
        return tp, fp, fn, tn

    def update(self, pred, true):
        pred, true = _validate_arrays(pred, true)
        if min(pred.shape[-2:]) < max(self.pool_sizes):
            raise ValueError(f"Spatial dimensions must be at least {max(self.pool_sizes)} for pooled CSI.")
        frame_shape = tuple(pred.shape[1:])
        if self._frame_shape is not None and frame_shape != self._frame_shape:
            raise ValueError("All batches must share the same [T,1,H,W] shape.")
        predicted_values = np.clip(pred * self.scale + self.offset, 0, self.max_val)
        observed_values = np.clip(true * self.scale + self.offset, 0, self.max_val)
        for index, threshold in enumerate(self.thresholds):
            predicted = predicted_values >= threshold
            observed = observed_values >= threshold
            for pool in self.pool_sizes:
                p_map = _apply_spatial_tolerance(predicted, pool)
                t_map = _apply_spatial_tolerance(observed, pool)
                chunk = self._confusion(p_map, t_map)
                self.counts[pool][index] = [
                    total + value for total, value in zip(self.counts[pool][index], chunk)
                ]
        self.sample_count += int(pred.shape[0])
        self.batch_count += 1
        self._frame_shape = frame_shape
        return self

    def compute(self):
        if self.sample_count == 0:
            raise ValueError("No evaluation samples have been accumulated.")
        result = {}
        csi, hss, bias, pool4, pool16 = [], [], [], [], []
        for index, threshold in enumerate(self.thresholds):
            suffix = str(int(threshold)) if threshold > 1 else str(threshold).replace(".", "p")
            tp, fp, fn, tn = self.counts[1][index]
            result[f"csi_{suffix}"] = float(CSI(tp, fp, fn))
            result[f"hss_{suffix}"] = float(HSS(tp, fp, fn, tn))
            result[f"bias_{suffix}"] = BIAS(tp + fp, tp + fn)
            csi.append(result[f"csi_{suffix}"])
            hss.append(result[f"hss_{suffix}"])
            bias.append(result[f"bias_{suffix}"])
            for pool, values in ((4, pool4), (16, pool16)):
                if pool not in self.counts:
                    continue
                pooled_tp, pooled_fp, pooled_fn, _ = self.counts[pool][index]
                value = float(CSI(pooled_tp, pooled_fp, pooled_fn))
                result[f"csi_pool{pool}_{suffix}"] = value
                values.append(value)
        result["avg_csi"] = float(np.mean(csi))
        result["avg_hss"] = float(np.mean(hss))
        result["avg_bias"] = float(np.mean(bias))
        if pool4:
            result["avg_csi_pool4"] = float(np.mean(pool4))
        if pool16:
            result["avg_csi_pool16"] = float(np.mean(pool16))
        return result


def metric(pred, true, dataset_name="default", metrics=_DEFAULT_METRICS):
    """Evaluate predictions against ground-truth radar sequences."""
    pred, true = _validate_arrays(pred, true)
    metrics = _validate_metrics(metrics)
    use_radar = "radar_metrics" in metrics or "csi" in metrics or "hss" in metrics
    acc = RadarMetricAccumulator(dataset_name=dataset_name, pool_sizes=(1, 4, 16) if use_radar else (1,))
    if use_radar or "bias" in metrics:
        acc.update(pred, true)

    pred_eval = np.clip(pred * acc.scale + acc.offset, 0, acc.max_val)
    true_eval = np.clip(true * acc.scale + acc.offset, 0, acc.max_val)

    eval_res = {}
    if use_radar:
        eval_res.update(acc.compute())
    elif "bias" in metrics:
        eval_res.update({key: value for key, value in acc.compute().items()
                         if key.startswith("bias_") or key == "avg_bias"})
    if "ssim" in metrics:
        from skimage.metrics import structural_similarity as cal_ssim
        B, T = pred_eval.shape[:2]
        ssim_vals = [
            cal_ssim(pred_eval[b, t].squeeze(), true_eval[b, t].squeeze(), data_range=acc.max_val)
            for b in range(B) for t in range(T)
        ]
        eval_res["ssim"] = float(np.mean(ssim_vals))

    return eval_res, _format_log(eval_res)


def evaluate_predictions(pred, true, dataset_name="default", batch_size=8,
                         metrics=_DEFAULT_METRICS, progress=None):
    pred, true = _validate_arrays(pred, true, check_finite=False)
    metrics = _validate_metrics(metrics)
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    use_radar = "radar_metrics" in metrics or "csi" in metrics or "hss" in metrics
    acc = RadarMetricAccumulator(dataset_name, pool_sizes=(1, 4, 16) if use_radar else (1,))
    if use_radar and min(pred.shape[-2:]) < 16:
        raise ValueError("Spatial dimensions must be at least 16 for CSI-pool16.")
    if "ssim" in metrics:
        if min(pred.shape[-2:]) < 7:
            raise ValueError("Spatial dimensions must be at least 7 for SSIM.")
        from skimage.metrics import structural_similarity as cal_ssim

    ssim_sum = 0.0
    frame_count = 0
    starts = range(0, pred.shape[0], int(batch_size))
    if progress is not None:
        starts = progress(starts)
    for start in starts:
        pred_chunk, true_chunk = _validate_arrays(pred[start:start + batch_size], true[start:start + batch_size])
        if use_radar or "bias" in metrics:
            acc.update(pred_chunk, true_chunk)
        pred_eval = np.clip(pred_chunk * acc.scale + acc.offset, 0, acc.max_val)
        true_eval = np.clip(true_chunk * acc.scale + acc.offset, 0, acc.max_val)
        if "ssim" in metrics:
            for batch_index in range(pred_eval.shape[0]):
                for lead in range(pred_eval.shape[1]):
                    ssim_sum += float(cal_ssim(pred_eval[batch_index, lead, 0],
                                               true_eval[batch_index, lead, 0], data_range=acc.max_val))
                    frame_count += 1

    result = {}
    if use_radar:
        result.update(acc.compute())
    elif "bias" in metrics:
        result.update({key: value for key, value in acc.compute().items()
                       if key.startswith("bias_") or key == "avg_bias"})
    if "ssim" in metrics:
        result["ssim"] = ssim_sum / frame_count
    return result, _format_log(result)