class SmoothMeter:
    """Computes and stores the average and current value"""
    def __init__(self, length=50):
        self.length = length
        self.reset()

    def reset(self):
        self.val_list = []
        self.val = 0
        self.avg = 0
        self.sum = 0

    def update(self, val, n=1):
        self.val_list = self.val_list + [val] * n
        self.val_list = self.val_list[-self.length:]
        self.val = val
        self.sum = sum(self.val_list)
        self.avg = self.sum / len(self.val_list)