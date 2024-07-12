# %%

import torch
from torch.distributions import beta
from fancy_einsum import einsum
import torch.optim
from tqdm import tqdm
import seaborn as sns
from training_utils import LinePlot

# %%
device = "cuda:0"

k = 20
activation_dim = 768
feature_dim = 25000
alpha = k / feature_dim
beta_dist = beta.Beta(1, 1/alpha)
bsz = 1000
sigma = 0.1
kl_loss = torch.nn.KLDivLoss(reduction="none")

# %%

# generate approx-orthogonal directions
init_d = torch.randn((feature_dim, activation_dim)).to(device)
init_d = init_d / init_d.norm(dim=-1, keepdim=True)
true_d = torch.nn.Parameter(init_d)
true_d.requires_grad = True

lr = 1e-3
optimizer = torch.optim.SGD([true_d], lr, weight_decay=0)
for i in range(10000):
    optimizer.zero_grad()

    # for feat_idx in tqdm(range(true_d.shape[0])):

    # hard to take the absolute value but this probably also works
    loss = einsum("feata act, featb act ->", true_d, true_d) / feature_dim
    loss.backward()

    print("avg cosine sim:", loss)
    optimizer.step()

    with torch.no_grad():
        true_d.divide_(true_d.norm(dim=-1, keepdim=True))

# %%
betas = beta_dist.sample((feature_dim,)).to(device)

def sample_x(n_x, std):
    # [n_x, feature_dim]
    zs = (torch.rand((n_x, feature_dim)).to(device) < betas) * 1

    # [n_x, activation_dim]
    xs = []
    max_batch = 100
    n_batches = (n_x - 1) // max_batch + 1
    for j in range(n_batches):
        start = j * max_batch 
        end = (j + 1) * max_batch
        xs.append((zs[start:min(end, zs.shape[0])].unsqueeze(dim=-1) * true_d).sum(dim=-2))    
    xs = torch.cat(xs, dim=0)
    xs = xs + torch.randn_like(xs).to(device) * std
    return zs, xs

def sigmoid(x):
    return x.exp() / (1 + x.exp())

# %%
init_est_d = torch.randn_like(true_d).to(device)
init_est_e = torch.randn_like(true_d).to(device)

init_est_d = init_est_d / init_est_d.norm(dim=-1, keepdim=True)
# init_est_e = init_est_e / init_est_e.norm(dim=-1, keepdim=True)

est_e = torch.nn.Parameter(init_est_e)
est_d = torch.nn.Parameter(init_est_d)

lr = 1e-3
optimizer = torch.optim.AdamW([est_e, est_d], lr=lr, weight_decay=0)

lp = LinePlot(['likelihood', 'kl_div'])
record_every = 100

for n_batches in tqdm(range(10000)):
    optimizer.zero_grad()

    evts = []
    evts.append(torch.cuda.Event(enable_timing=True))
    evts.append(torch.cuda.Event(enable_timing=True))

    evts[0].record()
    
    true_z, x = sample_x(bsz, sigma)

    evts[1].record()

    torch.cuda.synchronize()
    print(evts[0].elapsed_time(evts[1]))

    posterior = sigmoid(einsum("batch act, feat act -> batch feat", x, est_e))
    feature_sim = einsum("batch act, feat act -> batch feat", x, est_d)

    ortho_likelihood = (posterior * (feature_sim - 1).square() + (1 - posterior) * feature_sim.square()).sum()

    kl_div = posterior * (posterior / alpha).log() + (1 - posterior) * (torch.where(
        posterior == 1,
        1,
        1-posterior
    ) / (1-alpha)).log()
    kl_div = kl_div.sum()

    # kl_div = kl_loss(
    #     torch.tensor([alpha, 1-alpha]).to(device).log(),
    #     torch.stack([posterior, 1-posterior], dim=-1)
    # ).sum()

    loss = ortho_likelihood + kl_div
    loss.backward()
    optimizer.step()

    lp.add_entry({'likelihood': ortho_likelihood.item(), 
                  'kl_div': kl_div.item()
                  })

    with torch.no_grad():
        est_d.divide_(est_d.norm(dim=-1, keepdim=True))

    if n_batches % (-1 * record_every) == -1:
        lp.plot(start=0)
    
    if ortho_likelihood.isnan().sum() > 0 or kl_div.isnan().sum() > 0:
        break
# %%
