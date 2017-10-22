import torch
from torch.autograd.function import InplaceFunction


class Zoneout(InplaceFunction):
    r"""During training an RNN, randomly swaps some of the elements of the 
    input tensor with its values from a prevous time-step with probability *p* 
    using samples from a bernoulli distribution. The elements to be swapped are 
    randomized on every time-step.
    
    Zoneout is a variant of dropout (Hinton et al., 2012) designed specifically
    for regularizing recurrent connections of LSTMs or GRUs. While dropout
    applies a zero mask, zoneout applies an identity mask

    This has proven to be an effective technique for regularization of LSTMs 
    and GRUs as, contrary to dropout, gradient information and state 
    information are more readily propagated through time. For further 
    information, consult the paper
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activation`_ .

    Similarly to dropout, during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.15.
        inplace: If set to True, will do this operation in-place. Default: 
        False
        training: True if in training phase, False otherwise. Default: False.

    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> m = nn.Zoneout(p=0.25)
        >>> current_hidden_state = autograd.Variable(torch.Tensor([1, 2, 3])
        >>> previous_hidden_state = autograd.Variable(torch.Tensor([4, 5, 6])
        >>> output = m(current_hidden_state, previous_hidden_state)

    .. _Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activation
     https://arxiv.org/abs/1606.01305
    """

    def __init__(self, p=0.15, train=False, inplace=False, mask=None):
        super(Zoneout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("zoneout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.train = train
        self.inplace = inplace
        self.mask = mask

    def _make_noise(self, input):
        return input.new().resize_as_(input)

    def forward(self, current_input, previous_input):
        assert current_input.size() == previous_input.size(), \
            'Current and previous inputs must be of the same size, but ' \
            'current has size {current} and previous has size ' \
            '{previous}.'.format(
                current='x'.join(str(size) for size in current_input.size()),
                previous='x'.join(str(size) for size in previous_input.size())
            )
        if self.inplace:
            self.mark_dirty(current_input)
        else:
            current_input = current_input.clone()

        self.current_mask = self._make_noise(current_input)
        self.previous_mask = self._make_noise(previous_input)

        if self.train:
            if self.mask is not None:
                self.current_mask = self.mask
            else:
                self.current_mask.bernoulli_(1 - self.p)
            self.previous_mask.fill_(1).sub_(self.current_mask)
            output = (current_input * self.current_mask) + \
                     (previous_input * self.previous_mask)
        else:
            output = current_input
            self.current_mask.fill_(1)
            self.current_mask.fill_(0)

        return output

    def backward(self, grad_output):
        return grad_output * self.current_mask, \
               grad_output * self.previous_mask


def zoneout(current_input, previous_input, p=0.15, training=False, inplace=False, mask=None):
    return Zoneout(p, training, inplace, mask)(current_input, previous_input)

