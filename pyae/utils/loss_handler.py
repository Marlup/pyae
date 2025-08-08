# loss_handlers.py

import torch

class BaseLossHandler:
    def __init__(self, manager):
        self.manager = manager

    def compute(self, batch):
        raise NotImplementedError


class StandardLossHandler(BaseLossHandler):
    def compute(self, batch):
        x, y = batch["x"], batch["y"]
        x_cats = batch.get("x_categories")
        if self.manager.device:
            x, y = x.to(self.manager.device), y.to(self.manager.device)
            if x_cats:
                x_cats = [xc.to(self.manager.device) for xc in x_cats]
        outputs = self.manager.model(x, x_cats)
        return self.manager.criterion(outputs, y)


class StackLossHandler(BaseLossHandler):
    def compute(self, batch):
        x = batch["x"]
        if self.manager.device:
            x = x.to(self.manager.device)
        outputs, target = self.manager.model(x)
        return self.manager.criterion(outputs, target)


class VAELossHandler(BaseLossHandler):
    def compute(self, batch):
        x, y = batch["x"], batch["y"]
        if self.manager.device:
            x, y = x.to(self.manager.device), y.to(self.manager.device)
        outputs, z, mu, logvar = self.manager.model(x)
        return self.manager.criterion(outputs, y, mu, logvar)


class VMAELossHandler(BaseLossHandler):
    def compute(self, batch):
        x, y = batch["x"], batch["y"]
        if self.manager.device:
            x, y = x.to(self.manager.device), y.to(self.manager.device)
        out, map_out, z, map_z, mu, map_mu, logvar, map_logvar = self.manager.model(x)
        return self.manager.criterion(out, map_out, y, mu, map_mu, logvar, map_logvar)


class DCECLossHandler(BaseLossHandler):
    def compute(self, batch):
        x, y = batch["x"], batch["y"]
        if self.manager.device:
            x, y = x.to(self.manager.device), y.to(self.manager.device)
        outputs, z, q_dist = self.manager.model(x)
        if self.manager._should_update_p_target():
            self.manager._update_p_target()
        return self.manager.criterion(outputs, y, q_dist, self.manager.p_target)


class FactorVAELossHandler(BaseLossHandler):
    def compute(self, batch):
        x = batch["x"]
        if self.manager.device:
            x = x.to(self.manager.device)
        recon_x, z, logvar, mu = self.manager.model(x)
        logits_real = self.manager.discriminator(z)
        with torch.no_grad():
            z_perm = self.manager._permute_dims(z)
        logits_fake = self.manager.discriminator(z_perm)
        d_loss = self.manager.criterion_discriminator(logits_real, logits_fake)
        total_loss, *_ = self.manager.criterion(recon_x, x, mu, logvar, logits_real)
        return total_loss, d_loss


MODE_TO_HANDLER = {
    "standard": StandardLossHandler,
    "classification": StandardLossHandler,
    "stack": StackLossHandler,
    "vae": VAELossHandler,
    "vmae": VMAELossHandler,
    "dcec": DCECLossHandler,
    "factorVAE": FactorVAELossHandler,
}

def get_loss_handler(mode, manager):
    handler_cls = MODE_TO_HANDLER.get(mode)
    if handler_cls is None:
        raise ValueError(f"Unsupported mode: {mode}")
    return handler_cls(manager)