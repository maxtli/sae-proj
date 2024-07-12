
# %%
import torch
from training_utils import load_model_data
from sae_lens import SAE
from functools import partial
from fancy_einsum import einsum
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

# %%
sns.set()

# %%

kl_loss = torch.nn.KLDivLoss(reduction='none')
device="cuda"
model_name = "gpt2-small"
batch_size = 10
ctx_length = 25
top_features = 149
layer_no = 8
resid_hook = f"blocks.{layer_no}.hook_resid_pre"

# %%
device, model, tokenizer, owt_loader = load_model_data(model_name, batch_size, ctx_length)
# %%


sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = f"blocks.{layer_no}.hook_resid_pre", # won't always be a hook point
    device = "cpu"
)

sae = sae.to(device)

# %%

def projection_hook_last_token(similarity_storage, sae, act, hook):
    feature_vals = einsum("batch act, feature act -> batch feature", act[:,-1] - sae.b_dec, sae.W_dec)
    act_norms = act[:,-1].norm(dim=-1, keepdim=True)
    # [batch, n_features]
    vals, idx = feature_vals.topk(top_features, dim=-1)

    # [batch, n_features, seq_pos, d_model]
    act = act.unsqueeze(1).expand(-1,top_features,-1,-1).clone()
    act[:,:,-1] -= vals.unsqueeze(-1) * sae.W_dec[idx]
    act = act.flatten(start_dim=0,end_dim=1)

    similarity_storage['feature_val'] = vals
    similarity_storage['sim'] = vals / act_norms
    similarity_storage['feature_idx'] = idx
    return act

# %%

data = {"feature_val": [], "sim": [], "kl": [], "feature_idx": []}
all_prompts = []

with torch.no_grad():
    for b in tqdm(range(50)):
        batch = next(owt_loader)['tokens']
        all_prompts.append(tokenizer.batch_decode(batch))

        target_probs = model(batch)[:,-1].log_softmax(dim=-1)
        similarity_storage = {}
        probs = model.run_with_hooks(
            batch,
            fwd_hooks=[(resid_hook, partial(projection_hook_last_token, similarity_storage, sae))]
        )[:,-1].unflatten(0, (batch_size, top_features)).log_softmax(dim=-1)

        kl = kl_loss(probs, target_probs.unsqueeze(1).exp()).sum(dim=-1)

        data['kl'].append(kl)
        data['feature_val'].append(similarity_storage['feature_val'])
        data['sim'].append(similarity_storage['sim'])
        data['feature_idx'].append(similarity_storage['feature_idx'])

# %%
for k in data:
    data[k] = torch.cat(data[k], dim=0)
all_prompts = [x for b in all_prompts for x in b]

# %%
# sns.color_palette("viridis", as_cmap=True)
ax = sns.histplot(x=data['feature_val'].flatten().cpu(), y=data['kl'].log().flatten().cpu(), cbar = True, norm=LogNorm(), vmin=None, vmax=None, cmap="Spectral")
ax.set(xlabel="feature values", ylabel="importance")
# plt.yscale('log')

# %%
ax = sns.histplot(x=data['sim'].flatten().cpu(), y=data['kl'].log().flatten().cpu(), cbar = True, norm=LogNorm(), vmin=None, vmax=None, cmap="Spectral")
ax.set(xlabel="cos-sim", ylabel="importance")

# %%

ax = sns.histplot(x=data['sim'].flatten().cpu(), y=(data['kl'].log() - data['kl'].log().mean(dim=-1, keepdim=True)).flatten().cpu(), cbar = True, norm=LogNorm(), vmin=None, vmax=None, cmap="Spectral")
ax.set(xlabel="cos-sim", ylabel="importance vs mean")


# %%
_, kl_ranks = data['kl'].sort(dim=-1, descending=True)
_, sim_ranks = data['sim'].sort(dim=-1, descending=True)
ax = sns.histplot(x=sim_ranks.flatten().cpu(), y=kl_ranks.flatten().cpu(), cbar = True, norm=LogNorm(), vmin=None, vmax=None, binwidth=(1,1), cmap="Spectral")
ax.set(xlabel="cos-sim rank, 0=highest", ylabel="importance rank, 0=highest")

# %%
adv_egs = ((kl_ranks > 120) * (sim_ranks < 25)).nonzero()
pos_egs = ((kl_ranks > 120) * (sim_ranks > 120)).nonzero()

# %%

prompts = []
feature_descs = []

last_prompt = -1

for adv_eg in tqdm(pos_egs[:2000]):
    if adv_eg[0] == last_prompt:
        continue
    
    last_prompt = adv_eg[0]

    prompt = all_prompts[adv_eg[0]]
    feature_idx = data['feature_idx'][adv_eg[0], adv_eg[1]]
    r = requests.get(f"https://www.neuronpedia.org/api/feature/gpt2-small/{layer_no}-res-jb/{feature_idx}").json()['explanations'][0]['description']

    prompts.append(prompt)
    feature_descs.append(r)

# %%

for p in prompts:
    print(p.replace("\n", " "))
# %%

for d in feature_descs:
    print(d)
# %% 

# what about projecting to multiple directions? does it correlate?
# spot checking descriptive accuracy

# %%



[x[0] for x in sae.named_parameters()]

# %%

import requests

# %%
# %%
