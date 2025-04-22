class EarlyStopper:
    def __init__(self, tol=1e-4, max_no_improvements=5):
        self.tol = tol
        self.max_no_improvements = max_no_improvements
        self.prev_loss = float("inf")
        self.no_improvement_count = 0

    def reset(self):
        self.prev_loss = float("inf")
        self.no_improvement_count = 0

    def check(self, current_loss: float) -> bool:
        loss_change = abs(self.prev_loss - current_loss)

        if loss_change < self.tol:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0

        self.prev_loss = current_loss

        return self.no_improvement_count >= self.max_no_improvements
