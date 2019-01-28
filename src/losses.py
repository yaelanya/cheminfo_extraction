import torch

class SoftmaxLoss(torch.nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()

    def forward(self, forward_output, backward_output, targets):
        # target shape: (batch_size * timesteps,)
        if targets.dim() > 1:
            targets = targets.view(-1)

        # except PAD, BOS and EOS
        targets = targets[targets > 2]
        num_targets = torch.sum(targets > 2)

        forward_loss = torch.nn.functional.nll_loss(forward_output, targets, reduction='sum')
        backward_loss = torch.nn.functional.nll_loss(backward_output, targets, reduction='sum')

        average_loss = 0.5 * (forward_loss + backward_loss) / num_targets

        return average_loss
