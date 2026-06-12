"""Elman RNN case study from the PIV paper.

Task family: absorbing-state detection. For DFA state count k, the target
language is "the input contains a run of (k-1) consecutive 1s"; the minimal
DFA has exactly k states (trailing-run counter with an absorbing accept
state). k=3 (run of two 1s) is the development regime from the paper.

The hypothesised mechanism is an *input-invariant (latch) dimension*: a
hidden dimension that saturates after detection and stops responding to
input, implementing the absorbing state. The per-dimension input-invariance
score IV(d) (Eq. 1 of the paper) and the frozen threshold tau = 0.8
constitute the pre-registered certificate (see certificate.json and the
`certificate-freeze` git tag).

All populations are trained as one batched tensor program (leading dimension
= model index), so hundreds of models train in parallel on one device.

Requires torch (the only piv submodule that does).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "Population", "train_population", "persistence", "iv_scores",
    "certify", "weight_surgery", "matched_controls", "CERTIFICATE",
]

# Pre-registered certificate (frozen before blind-cohort generation; the
# freezing event is the `certificate-freeze` git tag on this repository).
CERTIFICATE = {
    "score": "max_d IV(d)",
    "iv_definition": "1 - mean_t Var_x(h_t[d] | post-detection) / Var_{x,t}(h_t[d] | all steps)",
    "threshold_tau": 0.8,
    "gap_epsilon": 0.2,
    "tight_dim_rule": "IV(d) >= 1 - gap_epsilon",
    "decision_rule": "certify iff max_d IV(d) >= threshold_tau",
    "eval_batch": 512,
    "eval_seq_len_factor": "5*(k-1), min 15",
}


# --------------------------------------------------------------------- tasks

def labels_for(x: torch.Tensor, k: int) -> torch.Tensor:
    """Per-timestep labels: 1 iff a run of (k-1) ones has occurred by t.

    x: [..., T] binary float tensor. Returns same-shape float labels.
    """
    run = torch.zeros_like(x[..., 0])
    fired = torch.zeros_like(x[..., 0], dtype=torch.bool)
    out = torch.empty_like(x)
    for t in range(x.shape[-1]):
        run = (run + 1) * x[..., t]
        fired = fired | (run >= k - 1)
        out[..., t] = fired.float()
    return out


def sample_batch(M: int, B: int, T: int, k: int, device,
                 gen: torch.Generator, plant: float = 0.0) -> torch.Tensor:
    """Random binary sequences [M, B, T]; optionally plant the target run
    (length k-1) in a `plant` fraction of each batch so positives exist at
    large k."""
    x = (torch.rand(M, B, T, generator=gen, device=device) < 0.5).float()
    if plant > 0:
        n_plant = int(B * plant)
        L = k - 1
        if n_plant > 0 and L <= T:
            pos = torch.randint(0, T - L + 1, (M, n_plant), generator=gen,
                                device=device)
            ar = torch.arange(L, device=device)
            idx = (pos.unsqueeze(-1) + ar).clamp(max=T - 1)        # [M,n,L]
            x[:, :n_plant, :].scatter_(2, idx, 1.0)
    return x


# -------------------------------------------------------------------- models

@dataclass
class Population:
    """A batch of M independent Elman RNNs (leading dim = model index)."""
    W_xh: torch.Tensor   # [M, H]
    W_hh: torch.Tensor   # [M, H, H]
    b_h: torch.Tensor    # [M, H]
    w_out: torch.Tensor  # [M, H]
    b_out: torch.Tensor  # [M]
    k: int = 3
    seeds: list = field(default_factory=list)

    @property
    def M(self) -> int:
        return self.W_hh.shape[0]

    @property
    def H(self) -> int:
        return self.W_hh.shape[1]

    def params(self):
        return [self.W_xh, self.W_hh, self.b_h, self.w_out, self.b_out]

    @staticmethod
    def init(seeds, H: int, k: int, device="cpu") -> "Population":
        """Per-model init from per-seed generators (U(-1/sqrt(H), 1/sqrt(H)))."""
        M = len(seeds)
        ws = {n: [] for n in ("W_xh", "W_hh", "b_h", "w_out", "b_out")}
        bound = 1.0 / math.sqrt(H)
        for s in seeds:
            g = torch.Generator().manual_seed(int(s))
            u = lambda *shape: (torch.rand(*shape, generator=g) * 2 - 1) * bound
            ws["W_xh"].append(u(H)); ws["W_hh"].append(u(H, H))
            ws["b_h"].append(u(H)); ws["w_out"].append(u(H))
            ws["b_out"].append(u(1).squeeze(0))
        t = {n: torch.stack(v).to(device).requires_grad_(True)
             for n, v in ws.items()}
        return Population(**t, k=k, seeds=[int(s) for s in seeds])

    def forward(self, x: torch.Tensor):
        """x: [M, B, T] -> (logits [M, B, T], hiddens [M, B, T, H])."""
        M, B, T = x.shape
        h = torch.zeros(M, B, self.H, device=x.device, dtype=x.dtype)
        logits, hs = [], []
        for t in range(T):
            pre = (x[:, :, t:t + 1] * self.W_xh.unsqueeze(1)
                   + torch.einsum("mbh,mgh->mbg", h, self.W_hh)
                   + self.b_h.unsqueeze(1))
            h = torch.tanh(pre)
            hs.append(h)
            logits.append(torch.einsum("mbh,mh->mb", h, self.w_out)
                          + self.b_out.unsqueeze(1))
        return torch.stack(logits, dim=2), torch.stack(hs, dim=2)

    def state_dict(self):
        return {"W_xh": self.W_xh.detach().cpu(), "W_hh": self.W_hh.detach().cpu(),
                "b_h": self.b_h.detach().cpu(), "w_out": self.w_out.detach().cpu(),
                "b_out": self.b_out.detach().cpu(), "k": self.k, "seeds": self.seeds}

    @staticmethod
    def load(d, device="cpu") -> "Population":
        return Population(
            W_xh=d["W_xh"].to(device), W_hh=d["W_hh"].to(device),
            b_h=d["b_h"].to(device), w_out=d["w_out"].to(device),
            b_out=d["b_out"].to(device), k=int(d["k"]), seeds=list(d["seeds"]))


# ------------------------------------------------------------------ training

def train_population(seeds, k: int = 3, H: int = 8, epochs: int = 10_000,
                     batch: int = 256, T: int | None = None, lr: float = 1e-3,
                     sat_lambda: float = 0.0, plant: float | None = None,
                     device: str = "cpu", data_seed: int = 0,
                     log_every: int = 0) -> tuple[Population, list[float]]:
    """Train M Elman RNNs in parallel with Adam (paper hyperparameters).

    sat_lambda > 0 is the *recipe*: an auxiliary post-detection saturation
    penalty  sat_lambda * mean[(1 - h^2) * post_mask]  that drives latch
    dimensions into the tanh saturation regime (gap suppression,
    Proposition 2 of the paper).
    """
    if T is None:
        T = max(15, 5 * (k - 1))
    if plant is None:
        plant = 0.0 if k <= 4 else 0.5
    pop = Population.init(seeds, H, k, device)
    opt = torch.optim.Adam(pop.params(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(data_seed)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    losses = []
    for ep in range(epochs):
        x = sample_batch(pop.M, batch, T, k, device, gen, plant)
        y = labels_for(x, k)
        logits, hs = pop.forward(x)
        loss = bce(logits, y)
        if sat_lambda > 0:
            mask = y.unsqueeze(-1)
            sat = ((1 - hs.pow(2)) * mask).sum() / mask.sum().clamp(min=1)
            loss = loss + sat_lambda * sat
        opt.zero_grad(); loss.backward(); opt.step()
        if log_every and (ep + 1) % log_every == 0:
            losses.append(float(loss))
            print(f"  epoch {ep+1}/{epochs}  loss {float(loss):.4f}", flush=True)
    return pop, losses


def val_loss(pop: Population, n: int = 512, data_seed: int = 10_000,
             device: str = "cpu") -> torch.Tensor:
    """Per-model BCE on a fresh held-out batch. Returns [M]."""
    T = max(15, 5 * (pop.k - 1))
    gen = torch.Generator(device=device).manual_seed(data_seed)
    plant = 0.0 if pop.k <= 4 else 0.5
    with torch.no_grad():
        x = sample_batch(pop.M, n, T, pop.k, device, gen, plant)
        y = labels_for(x, pop.k)
        logits, _ = pop.forward(x)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, reduction="none")
    return bce.mean(dim=(1, 2)).cpu()


# ---------------------------------------------------------------- evaluation

def persistence(pop: Population, trials: int = 1000, horizon: int = 500,
                data_seed: int = 7, device: str = "cpu",
                chunk: int = 250) -> torch.Tensor:
    """P(random post-detection trajectory stays correctly classified).

    Forces detection by planting the run at the start, then feeds `horizon`
    random symbols; a trial succeeds iff the output stays positive at every
    step. Returns per-model persistence [M].
    """
    k, H, M = pop.k, pop.H, pop.M
    gen = torch.Generator(device=device).manual_seed(data_seed)
    ok_total = torch.zeros(M, device=device)
    with torch.no_grad():
        for start in range(0, trials, chunk):
            B = min(chunk, trials - start)
            prefix = torch.ones(M, B, k - 1, device=device)
            tail = (torch.rand(M, B, horizon, generator=gen,
                               device=device) < 0.5).float()
            x = torch.cat([prefix, tail], dim=2)
            logits, _ = pop.forward(x)
            post = logits[:, :, k - 1:]              # outputs after detection
            ok_total += (post > 0).all(dim=2).float().sum(dim=1)
    return (ok_total / trials).cpu()


def iv_scores(pop: Population, n: int = 512, T: int | None = None,
              data_seed: int = 11, device: str = "cpu",
              eps: float = 1e-8) -> torch.Tensor:
    """Per-dimension input-invariance IV(d) (Eq. 1). Returns [M, H].

    IV(d) = 1 - mean_t Var_x(h_t[d]) over post-detection steps
                / Var_{x,t}(h_t[d]) over all steps.
    Models with no post-detection steps in the batch get IV = 0.
    """
    k = pop.k
    if T is None:
        T = max(15, 5 * (k - 1))
    gen = torch.Generator(device=device).manual_seed(data_seed)
    plant = 0.5  # ensure post-detection steps exist at every k
    with torch.no_grad():
        x = sample_batch(pop.M, n, T, k, device, gen, plant)
        y = labels_for(x, k)                      # [M,B,T]
        _, hs = pop.forward(x)                    # [M,B,T,H]
        mask = y.unsqueeze(-1)                    # post-detection indicator
        cnt_x = mask.sum(dim=1).clamp(min=1)                       # [M,T,1]
        mean_x = (hs * mask).sum(dim=1) / cnt_x                    # [M,T,H]
        var_x = ((hs - mean_x.unsqueeze(1)).pow(2) * mask).sum(dim=1) / cnt_x
        t_has = (y.sum(dim=1) > 1).unsqueeze(-1)                   # [M,T,1]
        num = (var_x * t_has).sum(dim=1) / t_has.sum(dim=1).clamp(min=1)
        den = hs.var(dim=(1, 2)) + eps                             # [M,H]
        iv = (1 - num / den).clamp(0, 1)
        iv = iv * (y.sum(dim=(1, 2)) > 0).float().unsqueeze(-1)
    return iv.cpu()


def certify(pop: Population, device: str = "cpu") -> dict:
    """Apply the frozen certificate. Returns dict of per-model tensors."""
    iv = iv_scores(pop, device=device)
    tau = CERTIFICATE["threshold_tau"]
    eps_gap = CERTIFICATE["gap_epsilon"]
    score = iv.max(dim=1).values
    return {
        "iv": iv,
        "score": score,
        "certified": score >= tau,
        "tight": iv >= 1 - eps_gap,            # [M, H] tight-dimension mask
        "n_tight": (iv >= 1 - eps_gap).sum(dim=1),
    }


# ------------------------------------------------------------- interventions

def weight_surgery(pop: Population, dim: int, alpha: float = 1.0,
                   mode: str = "row") -> Population:
    """Scale W_hh row (or column) `dim` by (1 - alpha) for every model."""
    W = pop.W_hh.detach().clone()
    if mode == "row":
        W[:, dim, :] *= (1 - alpha)
    else:
        W[:, :, dim] *= (1 - alpha)
    return Population(pop.W_xh.detach(), W, pop.b_h.detach(),
                      pop.w_out.detach(), pop.b_out.detach(), pop.k, pop.seeds)


def matched_controls(pop: Population, dim: int, alpha: float = 1.0,
                     seed: int = 13) -> tuple[Population, Population]:
    """Two matched controls for the targeted row surgery on `dim`:
    (i) random-direction row update with matched Frobenius norm;
    (ii) matched-rank Gaussian perturbation with matched norm.
    """
    g = torch.Generator().manual_seed(seed)
    W = pop.W_hh.detach()
    delta = alpha * W[:, dim, :]                       # removed component [M,H]
    norm = delta.norm(dim=1, keepdim=True)             # [M,1]
    # (i) random orthonormal direction, norm-matched, applied to a random row
    u = torch.randn(W.shape[0], W.shape[2], generator=g)
    u = (u / u.norm(dim=1, keepdim=True)).to(W.device) * norm
    W1 = W.clone()
    rows = torch.randint(0, W.shape[1], (W.shape[0],), generator=g)
    for m in range(W.shape[0]):
        W1[m, rows[m], :] -= u[m]
    # (ii) rank-1 Gaussian perturbation, norm-matched
    a = torch.randn(W.shape[0], W.shape[1], 1, generator=g).to(W.device)
    b = torch.randn(W.shape[0], 1, W.shape[2], generator=g).to(W.device)
    P = a * b
    P = P / P.flatten(1).norm(dim=1).view(-1, 1, 1) * norm.view(-1, 1, 1)
    W2 = W - P
    mk = lambda Wn: Population(pop.W_xh.detach(), Wn, pop.b_h.detach(),
                               pop.w_out.detach(), pop.b_out.detach(),
                               pop.k, pop.seeds)
    return mk(W1), mk(W2)


# ---------------------------------------------------------------- utilities

def manifest_rows(pop: Population, cohort: str, device: str = "cpu") -> list[dict]:
    """Per-model JSON rows: seed, persistence, certificate, val loss, params."""
    pers = persistence(pop, device=device)
    cert = certify(pop, device=device)
    vl = val_loss(pop, device=device)
    n_params = pop.H * pop.H + 3 * pop.H + 1
    rows = []
    for m in range(pop.M):
        rows.append({
            "cohort": cohort, "seed": pop.seeds[m], "k": pop.k, "H": pop.H,
            "persistence": round(float(pers[m]), 4),
            "is_generalizer": bool(pers[m] >= 0.99),
            "certificate_score": round(float(cert["score"][m]), 4),
            "certified": bool(cert["certified"][m]),
            "n_tight": int(cert["n_tight"][m]),
            "val_loss": round(float(vl[m]), 5),
            "param_count": n_params,
        })
    return rows


def save_artifacts(pop: Population, rows: list[dict], stem: str):
    torch.save(pop.state_dict(), f"{stem}.pt")
    with open(f"{stem}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
