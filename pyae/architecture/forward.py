def _compute_forward_loss(self, batch):
    x, y = batch["x"], batch["y"]
    if self.device:
        x, y = x.to(self.device), y.to(self.device)

    if self.mode in ("standard", "classification"):
        x_cats = batch.get("x_categories")
        if x_cats and self.device:
            x_cats = [xc.to(self.device) for xc in x_cats]
        outputs = self.model(x, x_cats)
        return self.criterion(outputs, y)

    elif self.mode == "stack":
        outputs, target = self.model(x)
        return self.criterion(outputs, target)

    elif self.mode == "vae":
        outputs, z, mu, logvar = self.model(x)
        return self.criterion(outputs, y, mu, logvar)

    elif self.mode == "vmae":
        out, map_out, z, map_z, mu, map_mu, logvar, map_logvar = self.model(x)
        return self.criterion(out, map_out, y, mu, map_mu, logvar, map_logvar)

    elif self.mode == "factorVAE":
        recon_x, z, logvar, mu = self.model(x)
        logits_real = self.discriminator(z)
        with torch.no_grad():
            z_perm = self._permute_dims(z)
        logits_fake = self.discriminator(z_perm)
        d_loss = self.criterion_discriminator(logits_real, logits_fake)
        total_loss, *_ = self.criterion(recon_x, x, mu, logvar, logits_real)
        return total_loss, d_loss

    else:
        raise ValueError(f"Unsupported mode: {self.mode}")
    
def _permute_dims(self, z: torch.Tensor) -> torch.Tensor:
    B, D = z.size()
    z_perm = [z[torch.randperm(B), d] for d in range(D)]
    return torch.stack(z_perm, dim=1)