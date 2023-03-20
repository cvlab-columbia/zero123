import os

class EarlyLoopBreak():
    def __init__(self, break_at: int):
        self.iter = 0
        self.break_at = break_at
        self.on = bool(os.environ.get("EBREAK"))

    def on_break(self):
        if not self.on:
            return

        self.iter += 1
        if self.break_at > 0 and self.iter >= self.break_at:
            return True
