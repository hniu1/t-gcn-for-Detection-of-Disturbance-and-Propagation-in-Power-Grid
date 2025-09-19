import numpy as np
from scipy import signal
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def build_signal_graph(sites, data, t_event, pre_window=120.0, guard=10.0,
                       k=6, lag_lambda=2.0):
    """
    Returns:
        W      : [N,N] symmetric weighted adjacency
        A_hat  : [N,N] normalized adjacency for GCN (D^-1/2 (W+I) D^-1/2)
    Args:
        t_event    : global event time (sec)
        pre_window : how much pre-event data to use for similarity (sec)
        guard      : exclude the last 'guard' seconds before event to avoid leakage
        k          : k-NN per node
        lag_lambda : seconds, controls how fast similarity decays with lag
    """
    N = len(sites)
    W = np.zeros((N, N), dtype=float)

    # Collect pre-event residuals aligned per site
    t_lo = max(0.0, t_event - pre_window)
    t_hi = max(0.0, t_event - guard)

    # Build an aligned matrix (pad with NaN then drop)
    series = []
    for sid in sites:
        df = data[sid]
        seg = df[(df['t'] >= t_lo) & (df['t'] <= t_hi)].copy()
        series.append(seg['r'].to_numpy())
    # To handle minor length mismatches, align by min length
    L = min(len(x) for x in series)
    X = np.stack([x[:L] for x in series], axis=0)  # [N, L]

    # Pairwise similarity + lag (event-time xcorr optional; here use pre-event only)
    for i in range(N):
        xi = X[i] - X[i].mean()
        for j in range(i+1, N):
            xj = X[j] - X[j].mean()
            # Pearson correlation
            denom = (xi.std() * xj.std()) + 1e-12
            rho = float(np.dot(xi, xj) / denom / L)

            # Cross-correlation for lag around zero (use small window)
            corr = signal.correlate(xi, xj, mode='full')
            lags = signal.correlation_lags(len(xi), len(xj), mode='full')
            lag_samples = lags[np.argmax(corr)]
            dt = data[sites[i]].attrs.get('dt', np.nan)
            lag_sec = float(lag_samples * dt) if np.isfinite(dt) else 0.0

            w = max(0.0, rho)**2 * np.exp(-abs(lag_sec) / lag_lambda)
            W[i, j] = W[j, i] = w

    # k-NN sparsify (keep top-k per row)
    for i in range(N):
        idx = np.argsort(W[i])[::-1]
        keep = idx[:k]
        mask = np.ones(N, dtype=bool); mask[keep] = False; mask[i] = False
        W[i, mask] = 0.0
    W = np.maximum(W, W.T)  # symmetrize again

    # Add self-loops and normalize for GCN
    A = W + np.eye(N)
    d = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(d, 1e-12, None)))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return W, A_hat

def stack_node_features(sites, data, feature_cols=('r', 'roc_ps')):
    """
    Aligns all sites on their exact DateTime index (your preprocess already
    puts them on the same 0.1s grid). Returns:
      tvec: np.array of relative seconds [T]
      X:    np.array [T, N, F] with features per node
    """
    # Use the first site's timeline as reference
    ref = data[sites[0]][['DateTime','t']].copy()
    ref.rename(columns={'t':'tref'}, inplace=True)

    feat_frames = []
    for sid in sites:
        df = data[sid][['DateTime'] + list(feature_cols)].copy()
        df.columns = ['DateTime'] + [f'{sid}:{c}' for c in feature_cols]
        feat_frames.append(df)

    M = ref
    for df in feat_frames:
        M = M.merge(df, on='DateTime', how='inner')

    # Extract t and features
    tvec = M['tref'].to_numpy()
    Fcols = [c for c in M.columns if ':' in c]   # all feature columns
    F = len(feature_cols)
    N = len(sites)
    assert len(Fcols) == N*F, "Feature assembly mismatch."

    # Order features as [site1:feat1, site1:feat2, site2:feat1, ...]
    ordered = []
    for sid in sites:
        for c in feature_cols:
            ordered.append(f'{sid}:{c}')
    X = M[ordered].to_numpy(dtype=np.float32).reshape(len(tvec), N, F)
    return tvec, X

# Build training windows for pre-event
def make_windows(X, tvec, lookback=60, horizon=1):
    """
    Returns start indices where a window [i-lookback+1 ... i] exists and
    we can predict next 'horizon' steps.
    """
    T = len(tvec)
    starts = np.arange(lookback-1, T - horizon, dtype=int)
    return starts

def filter_pre_event(starts, tvec, t_event, guard=10.0, lookback=60):
    """
    Keep only windows whose *entire* history ends before (t_event - guard)
    to avoid leakage from the event.
    """
    t_end   = tvec[starts]
    ok = t_end <= (t_event - guard)
    return starts[ok]

class SeqDataset(Dataset):
    def __init__(self, X, starts, lookback=60, horizon=1, predict_col=0):
        """
        X: [T, N, F]  (F includes ['r','roc_ps']; we predict 'r' = predict_col=0)
        """
        self.X = X
        self.starts = starts
        self.lookback = lookback
        self.horizon = horizon
        self.predict_col = predict_col

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        '''
        returns:
        xwin: [lookback, N, F] history (all nodes, both features)
        y: [horizon, N, 1] target (predicting feature 0, i.e., residual r)
        '''
        s = self.starts[idx]
        # history window [s-lookback+1, ..., s]  => shape [L, N, F]
        xwin = self.X[s-self.lookback+1 : s+1]              # [L,N,F]
        # target next step(s) residual 'r' => [H, N, 1]
        y = self.X[s+1 : s+1+self.horizon, :, self.predict_col:self.predict_col+1]
        return torch.from_numpy(xwin), torch.from_numpy(y)

class TGCN(nn.Module):
    def __init__(self, N, F_in=2, gcn_hidden=32, rnn_hidden=64, horizon=1, A_hat=None, dropout=0.1):
        super().__init__()
        self.N = N
        self.F_in = F_in
        self.gcn_hidden = gcn_hidden
        self.rnn_hidden = rnn_hidden
        self.horizon = horizon

        # Graph conv weights: W: [F_in, gcn_hidden]
        self.W = nn.Parameter(torch.randn(F_in, gcn_hidden) * 0.1)
        self.bias = nn.Parameter(torch.zeros(gcn_hidden))

        # GRU runs over time; treat nodes as batch
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=rnn_hidden, batch_first=False)
        self.dropout = nn.Dropout(dropout)

        # Head predicts next H residual steps per node
        self.head = nn.Linear(rnn_hidden, horizon)

        # register normalized adjacency
        if A_hat is None:
            raise ValueError("A_hat (normalized adjacency) is required")
        A = torch.tensor(A_hat, dtype=torch.float32)
        self.register_buffer('A_hat', A)

    def gcn_step(self, X_t):
        """
        X_t: [N, F_in]
        returns: [N, gcn_hidden]
        A_hat X_t W + b
        """
        AX = self.A_hat @ X_t               # [N, F_in]
        out = AX @ self.W + self.bias       # [N, gcn_hidden]
        return torch.relu(out)

    def forward(self, X_seq):
        """
        X_seq: [L, N, F_in]
        Returns: pred [horizon, N] of residual r
        """
        L, N, F = X_seq.shape
        assert N == self.N and F == self.F_in

        # Apply GCN at each timestep
        g_list = []
        for t in range(L):
            g_list.append(self.gcn_step(X_seq[t]))  # [N, g]
        G = torch.stack(g_list, dim=0)              # [L, N, g]

        # GRU expects [L, batch, feat], where batch = N (nodes)
        G = self.dropout(G)
        out, h = self.gru(G)                        # h: [1, N, rnn_hidden]
        h_last = h[0]                               # [N, rnn_hidden]

        # Per node head → [N, H]
        y = self.head(h_last)                       # [N, H]
        # Return as [H, N] for convenience
        return y.transpose(0,1)
    
def train_tgcn(X, tvec, A_hat, t_event, lookback=60, horizon=1,
               lr=1e-3, weight_decay=1e-4, batch_size=64, epochs=20, guard=10.0, device='cpu'):
    """
    X: [T,N,F] numpy
    """
    T, N, F = X.shape
    starts_all = make_windows(X, tvec, lookback, horizon)
    starts_trn = filter_pre_event(starts_all, tvec, t_event, guard=guard, lookback=lookback)

    ds = SeqDataset(X, starts_trn, lookback, horizon, predict_col=0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TGCN(N=N, F_in=F, gcn_hidden=32, rnn_hidden=64, horizon=horizon, A_hat=A_hat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=0.02)  # robust to small spikes

    model.train()
    for ep in range(1, epochs+1):
        total = 0.0; count = 0
        for xb, yb in dl:
            xb = xb.to(device)          # [B,L,N,F]
            yb = yb.to(device)          # [B,H,N,1]
            # collapse batch into time-major by stacking nodes as batch
            # Our model expects [L,N,F]; so we average across batch after forward.
            # We can process each item in batch; but vectorizing is simpler by stacking L dims.
            # Here we loop (batch is small).
            loss = 0.0
            for b in range(xb.shape[0]):
                pred = model(xb[b])     # [H, N]
                # target residual r only
                tgt = yb[b, :, :, 0]    # [H, N]
                loss = loss + loss_fn(pred, tgt)
            loss = loss / xb.shape[0]

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); count += 1
        print(f"epoch {ep:02d} | loss {total/max(1,count):.6f}")
    return model

def score_full_day(model, X, tvec, lookback=60, horizon=1, dt=0.1, alpha=0.3):
    """
    Returns:
      err_r:   [T, N]   one-step residual abs error
      z:       [T, N]   z-score per node (fitted on pre-event region later)
      pred_r:  [T, N]   one-step predictions aligned at t+1 for each window end
    """
    model.eval()
    T, N, F = X.shape
    starts = make_windows(X, tvec, lookback, horizon)
    pred_all = np.full((T, N), np.nan, dtype=np.float32)
    err_all  = np.full((T, N), np.nan, dtype=np.float32)

    with torch.no_grad():
        for s in starts:
            xwin = torch.from_numpy(X[s-lookback+1:s+1]).to(next(model.parameters()).device)
            pred = model(xwin).cpu().numpy()  # [H,N]
            # place the 1-step prediction at time index s+1
            pred1 = pred[0]                   # [N]
            r_true = X[s+1, :, 0]             # true residual r
            df_true = X[s+1, :, 1]            # true RoCoF (optional)
            # simple combined error (|r_error| + alpha*|df_error| if you forecast df too)
            e = np.abs(pred1 - r_true)
            pred_all[s+1, :] = pred1
            err_all[s+1, :]  = e

    # z-score per node using pre-event portion (caller should slice)
    z = np.zeros_like(err_all)
    # We'll let caller compute node-wise mean/std on their chosen baseline.
    return err_all, z, pred_all

def fit_nodewise_z(err, tvec, t_event, guard=10.0, min_baseline_s=5.0):
    """
    Fit mean/std per node on baseline (t <= t_event - guard).
    If that baseline is too short/empty, fall back to the earliest window
    before t_event with at least min_baseline_s of data; if still short,
    use robust stats (median/MAD) over all pre-event samples.
    """
    err = np.asarray(err, float)
    dt = float(np.median(np.diff(tvec))) if len(tvec) > 1 else 0.1

    # Primary baseline
    baseline = (tvec <= (t_event - guard))
    if baseline.sum() < max(3, int(np.ceil(min_baseline_s / max(dt, 1e-9)))):
        # Fallback: earliest window before event with length >= min_baseline_s
        pre_mask = (tvec < t_event - 1e-6)
        if pre_mask.any():
            # take the first K points of pre-event if long enough, else all pre-event
            K = int(np.ceil(min_baseline_s / max(dt, 1e-9)))
            idx = np.where(pre_mask)[0]
            baseline = np.zeros_like(pre_mask, dtype=bool)
            baseline[idx[:min(len(idx), K)]] = True
        else:
            baseline = np.zeros_like(pre_mask, dtype=bool)

    # If still empty, robust stats over everything as a last resort
    if baseline.sum() == 0:
        mu = np.nanmedian(err, axis=0)
        mad = np.nanmedian(np.abs(err - mu[None, :]), axis=0)
        sd = 1.4826 * mad
    else:
        mu = np.nanmean(err[baseline], axis=0)
        sd = np.nanstd (err[baseline], axis=0)

    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1e-6)
    z = (err - mu[None, :]) / sd[None, :]
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z, mu, sd


def arrival_times(z, tvec, tau=2.5, persist_s=2.0, dt=0.1):
    """
    Arrival time per node = first t where z>tau for at least persist_s.
    Returns array [N] with np.nan if never exceeds.
    """
    N = z.shape[1]
    Lp = max(1, int(round(persist_s / dt)))
    T = len(tvec)
    T_i = np.full(N, np.nan, dtype=float)
    for j in range(N):
        seq = np.nan_to_num(z[:, j], nan=0.0)
        above = (seq > tau).astype(np.int32)
        # rolling sum of length Lp
        roll = np.convolve(above, np.ones(Lp, dtype=np.int32), mode='same')
        idx = np.where(roll >= Lp)[0]
        if len(idx) > 0:
            T_i[j] = tvec[idx[0]]
    return T_i