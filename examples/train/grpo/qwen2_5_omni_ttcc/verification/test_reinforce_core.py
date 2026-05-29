"""Unit tests for reinforce_core.py (Gate 2 math, no GPU). Verifies the DANGEROUS
head-PG math in isolation: log-prob correctness, monotone curve, reward/advantage
DETACHED (grad only via mu_z), gradient DIRECTION, KL>=0 pulling to ref.
Run: python test_reinforce_core.py
"""
import math
import torch
import reinforce_core as R

torch.manual_seed(0)
fails = []
def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond: fails.append(name)

print("=== T1 gaussian_logp matches torch.distributions ===")
mu = torch.randn(4, 8); z = torch.randn(4, 8); sigma = 0.7
ref = torch.distributions.Normal(mu, sigma).log_prob(z).sum(-1)
ours = R.gaussian_logp(z, mu, sigma)
check("logp == Normal.log_prob.sum", torch.allclose(ours, ref, atol=1e-5),
      f"(max diff {(ours-ref).abs().max():.2e})")

print("=== T2 curve is monotone, R(0)=1, in [0,1] for ANY hazards ===")
z = torch.randn(100, 60) * 5  # extreme
curve = R.curve_from_hazards(z)
check("R(0)==1", torch.allclose(curve[:, 0], torch.ones(100)))
check("monotone non-increasing", bool((curve[:, 1:] <= curve[:, :-1] + 1e-6).all()))
check("in [0,1]", bool((curve >= 0).all() and (curve <= 1.0 + 1e-6).all()))

print("=== T3 group_advantage: zero-mean per group ===")
rew = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])  # 2 groups of 3
adv = R.group_advantage(rew, group_size=3)
g0 = adv[:3]
check("group0 zero-mean", abs(float(g0.mean())) < 1e-5)
check("group1 (constant) -> ~0 adv", bool((adv[3:].abs() < 1e-3).all()), f"({adv[3:].tolist()})")

print("=== T4 reinforce_loss: reward/advantage/action DETACHED (grad only via mu_z) ===")
mu = torch.zeros(1, 4, requires_grad=True)
z = (mu + 0.7 * torch.randn(1, 4)).detach()           # sampled action (detached)
adv = torch.tensor([1.0])
loss = R.reinforce_loss(mu, z, adv, sigma=0.7)
loss.backward()
check("grad flows to mu_z", mu.grad is not None and torch.isfinite(mu.grad).all())
# reward path detach check: a "reward" computed from mu must NOT contribute grad
mu2 = torch.zeros(1, 4, requires_grad=True)
z2 = (mu2 + 0.7*torch.randn(1,4)).detach()
fake_reward = R.curve_from_hazards(mu2).sum()          # depends on mu2 (would leak if not detached)
adv2 = R.group_advantage(torch.stack([fake_reward, fake_reward*0]).view(-1), 2)  # uses reward
loss2 = R.reinforce_loss(mu2, z2, adv2, sigma=0.7)
g = torch.autograd.grad(loss2, mu2, retain_graph=True)[0]
# analytic grad if advantage truly detached: d/dmu [-(logp(z2;mu)*adv.detach())]
logp = R.gaussian_logp(z2, mu2, 0.7); analytic = torch.autograd.grad(-(logp*adv2.detach()).mean(), mu2)[0]
check("loss grad == grad with advantage detached (no reward leak)", torch.allclose(g, analytic, atol=1e-6))

print("=== T5 gradient DIRECTION: good sample (A>0) pulls mu toward it ===")
mu = torch.zeros(1, 1, requires_grad=True)
z_high = torch.tensor([[2.0]])    # sample above mu
loss = R.reinforce_loss(mu, z_high, torch.tensor([1.0]), sigma=1.0)  # A>0 (good)
gmu = torch.autograd.grad(loss, mu)[0]
check("z'>mu & A>0 -> grad<0 (descent raises mu toward z')", float(gmu) < 0, f"(grad={float(gmu):.3f})")
mu = torch.zeros(1,1, requires_grad=True)
loss_bad = R.reinforce_loss(mu, z_high, torch.tensor([-1.0]), sigma=1.0)  # A<0 (bad)
gmu_bad = torch.autograd.grad(loss_bad, mu)[0]
check("z'>mu & A<0 -> grad>0 (descent lowers mu away)", float(gmu_bad) > 0, f"(grad={float(gmu_bad):.3f})")

print("=== T6 KL>=0 and pulls mu toward ref ===")
mu = torch.tensor([[1.0, -1.0]], requires_grad=True); ref = torch.tensor([[0.0, 0.0]])
kl = R.kl_to_ref(mu, ref, sigma=0.7)
check("KL>=0", float(kl) >= 0, f"(={float(kl):.3f})")
gkl = torch.autograd.grad(kl, mu)[0]
check("KL grad pulls mu toward ref (sign(grad)==sign(mu-ref))",
      bool((torch.sign(gkl) == torch.sign(mu.detach()-ref)).all()))

print()
if fails:
    print(f"==== {len(fails)} FAILED: {fails} ===="); raise SystemExit(1)
print("==== ALL REINFORCE-CORE TESTS PASSED ====")
