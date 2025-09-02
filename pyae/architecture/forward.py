
def compute_forward_loss(batch, model, criterion, mode):
    return _do_compute_forward_loss(batch, model, criterion, mode)

def _do_compute_forward_loss(batch, model, criterion, mode="standard"):
    x, y = batch["x"], batch["y"]

    if mode in ("standard", "classification"):
        x_cats = batch.get("x_categories")
        outputs = model(x, x_cats)
        return criterion(outputs, y)

    elif mode == "stack":
        outputs, target = model(x)
        return criterion(outputs, target)

    elif mode == "vae":
        outputs, z, mu, logvar = model(x)
        return criterion(outputs, y, mu, logvar)

    elif mode == "vmae":
        out, map_out, z, map_z, mu, map_mu, logvar, map_logvar = model(x)
        return criterion(out, map_out, y, mu, map_mu, logvar, map_logvar)

    elif mode == "factorVAE":
        recon_x, z, logvar, mu = model(x)
        logits_real = discriminator(z)
        with torch.no_grad():
            z_perm = _permute_dims(z)
        logits_fake = discriminator(z_perm)
        d_loss = criterion_discriminator(logits_real, logits_fake)
        total_loss, *_ = criterion(recon_x, x, mu, logvar, logits_real)
        return total_loss, d_loss

    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
def _permute_dims(self, z: torch.Tensor) -> torch.Tensor:
    B, D = z.size()
    z_perm = [z[torch.randperm(B), d] for d in range(D)]
    return torch.stack(z_perm, dim=1)

# implementar ()

# (1)
#     if device:
#         x, y = x.to(device), y.to(device)

# (2)
# x_cats = [xc.to(device) for xc in x_cats]