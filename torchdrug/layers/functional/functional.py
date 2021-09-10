import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch_scatter.composite import scatter_log_softmax, scatter_softmax
from torch.nn import functional as F


def multinomial(input, num_sample, replacement=False):
    """
    Fast multinomial sampling. This is the default implementation in PyTorch v1.6.0+.

    Parameters:
        input (Tensor): unnormalized distribution
        num_sample (int): number of samples
        replacement (bool, optional): sample with replacement or not
    """
    if replacement:
        return torch.multinomial(input, num_sample, replacement)

    rand = torch.rand_like(input).log() / input
    samples = rand.topk(num_sample).indices
    return samples


def masked_mean(input, mask, dim=None, keepdim=False):
    """
    Masked mean of a tensor.

    Parameters:
        input (Tensor): input tensor
        mask (BoolTensor): mask tensor
        dim (int or tuple of int, optional): dimension to reduce
        keepdim (bool, optional): whether retain ``dim`` or not
    """
    input = input.masked_scatter(~mask, torch.zeros_like(input)) # safe with nan
    if dim is None:
        return input.sum() / mask.sum().clamp(1)
    return input.sum(dim, keepdim=keepdim) / mask.sum(dim, keepdim=keepdim).clamp(1)


def mean_with_nan(input, dim=None, keepdim=False):
    """
    Mean of a tensor. Ignore all nan values.

    Parameters:
        input (Tensor): input tensor
        dim (int or tuple of int, optional): dimension to reduce
        keepdim (bool, optional): whether retain ``dim`` or not
    """
    mask = ~torch.isnan(input)
    return masked_mean(input, mask, dim, keepdim)


def shifted_softplus(input):
    """
    Shifted softplus function.

    Parameters:
        input (Tensor): input tensor
    """
    return F.softplus(input) - F.softplus(torch.zeros(1, device=input.device))


def multi_slice_mask(starts, ends, length):
    """
    Compute the union of multiple slices into a binary mask.

    Example::

        >>> mask = F.multi_slice_mask(torch.tensor([0, 1, 4]), torch.tensor([2, 3, 6]), 6)
        >>> assert (mask == torch.tensor([1, 1, 1, 0, 1, 1])).all()

    Parameters:
        starts (LongTensor): start indexes of slices
        ends (LongTensor): end indexes of slices
        length (int): length of mask
    """
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    indexes = torch.cat([starts, ends])
    if indexes.numel():
        assert indexes.min() >= 0 and indexes.max() <= length
    mask = scatter_add(values, indexes, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def as_mask(indexes, length):
    """
    Convert indexes into a binary mask.

    Parameters:
        indexes (LongTensor): positive indexes
        length (int): maximal possible value of indexes
    """
    mask = torch.zeros(length, dtype=torch.bool, device=indexes.device)
    mask[indexes] = 1
    return mask


def _size_to_index(size):
    """
    Convert sizes to variadic indexes.

    Example::

        >>> index = _size_to_index(torch.tensor([3, 2, 1]))
        >>> assert (index == torch.tensor([0, 0, 0, 1, 1, 2])).all()

    Parameters:
        size (LongTensor): size of each sample
    """
    cum_size = size.cumsum(0)
    # special case 1: size[-1] = 0
    index = cum_size[cum_size < cum_size[-1]]
    # special case 2: size[i] = size[i+1] = 0
    index2sample = scatter_add(torch.ones_like(index), index, dim_size=cum_size[-1])
    index2sample = index2sample.cumsum(0)
    return index2sample


def _extend(data, size, input, input_size):
    """
    Extend variadic-sized data with variadic-sized input.
    This is a variadic variant of ``torch.cat([data, input], dim=-1)``.

    Example::

        >>> data = torch.tensor([0, 1, 2, 3, 4])
        >>> size = torch.tensor([3, 2])
        >>> input = torch.tensor([-1, -2, -3])
        >>> input_size = torch.tensor([1, 2])
        >>> new_data, new_size = _extend(data, size, input, input_size)
        >>> assert (new_data == torch.tensor([0, 1, 2, -1, 3, 4, -2, -3])).all()
        >>> assert (new_size == torch.tensor([4, 4])).all()

    Parameters:
        data (Tensor): variadic data
        size (LongTensor): size of data
        input (Tensor): variadic input
        input_size (LongTensor): size of input

    Returns:
        (Tensor, LongTensor): output data, output size
    """
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size


def variadic_sum(input, size):
    """
    Compute sum over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`

    Returns
        Tensor: sum
    """
    index2sample = _size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    value = scatter_add(input, index2sample, dim=0)
    return value


def variadic_mean(input, size):
    """
    Compute mean over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`

    Returns
        Tensor: mean
    """
    index2sample = _size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    value = scatter_mean(input, index2sample, dim=0)
    return value


def variadic_max(input, size):
    """
    Compute max over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`

    Returns
        (Tensor, LongTensor): max values and indexes
    """
    index2sample = _size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    value, index = scatter_max(input, index2sample, dim=0)
    index = index - size.cumsum(0) + size
    return value, index


def variadic_log_softmax(input, size):
    """
    Compute log softmax over categories with variadic sizes.

    Suppose there are :math:`N` samples, and the numbers of categories in all samples are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): number of categories of shape :math:`(N,)`
    """
    index2sample = _size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    log_likelihood = scatter_log_softmax(input, index2sample, dim=0)
    return log_likelihood


def variadic_softmax(input, size):
    """
    Compute softmax over categories with variadic sizes.

    Suppose there are :math:`N` samples, and the numbers of categories in all samples are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): number of categories of shape :math:`(N,)`
    """
    index2sample = _size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    log_likelihood = scatter_softmax(input, index2sample, dim=0)
    return log_likelihood


def variadic_cross_entropy(input, target, size, reduction="mean"):
    """
    Compute cross entropy loss over categories with variadic sizes.

    Suppose there are :math:`N` samples, and the numbers of categories in all samples are summed to :math:`B`.

    Parameters:
        input (Tensor): prediction of shape :math:`(B, ...)`
        target (Tensor): target of shape :math:`(N, ...)`. Each target is a relative index in a sample.
        size (LongTensor): number of categories of shape :math:`(N,)`
        reduction (string, optional): reduction to apply to the output.
            Available reductions are ``none``, ``sum`` and ``mean``.
    """
    index2sample = _size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    log_likelihood = scatter_log_softmax(input, index2sample, dim=0)
    size = size.view([-1] + [1] * (input.ndim - 1))
    assert (target >= 0).all() and (target < size).all()
    target_index = target + size.cumsum(0) - size
    loss = -log_likelihood.gather(0, target_index)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("Unknown reduction `%s`" % reduction)


def variadic_topk(input, size, k, largest=True):
    """
    Compute the :math:`k` largest elements over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    If any set has less than than :math:`k` elements, the size-th largest element will be
    repeated to pad the output to :math:`k`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
        k (int): the k in "top-k"
        largest (bool, optional): return largest or smallest elements

    Returns
        (Tensor, LongTensor): top-k values and indexes
    """
    index2graph = _size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if largest:
        offset = -offset
    input_ext = safe_input + offset * index2graph
    index_ext = input_ext.argsort(dim=0, descending=largest)
    num_actual = size.clamp(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()

    if (num_padding > 0).any():
        # special case: size < k, pad with the last valid index
        padding = ends - 1
        padding2graph = _size_to_index(num_padding)
        mask = _extend(mask, num_actual, padding[padding2graph], num_padding)[0]

    index = index_ext[mask] # (N * k, ...)
    value = input.gather(0, index)
    value = value.view(-1, k, *input.shape[1:])
    index = index.view(-1, k, *input.shape[1:])
    index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index


def variadic_sort(input, size, descending=False):
    """
    Sort elements in sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
        descending (bool, optional): return ascending or descending order
    """
    index2sample = _size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if descending:
        offset = -offset
    input_ext = safe_input + offset * index2sample
    index = input_ext.argsort(dim=0, descending=descending)
    value = input.gather(0, index)
    index = index - (size.cumsum(0) - size)[index2sample]
    return value, index


def variadic_arange(size, device=None):
    """
    Return a 1-D tensor that contains integer intervals of variadic sizes.
    This is a variadic variant of ``torch.arange(stop).expand(batch_size, -1)``.

    Suppose there are :math:`N` intervals.

    Parameters:
        size (LongTensor): size of intervals of shape :math:`(N,)`
        device (torch.device, optional): device of the tensor
    """
    index2sample = _size_to_index(size)
    starts = size.cumsum(0) - size

    range = torch.arange(size.sum(), device=device)
    range = range - starts[index2sample]
    return range


def variadic_randperm(size, device=None):
    """
    Return random permutations for sets with variadic sizes.
    The ``i``-th permutation contains integers from 0 to ``size[i] - 1``.

    Suppose there are :math:`N` sets.

    Parameters:
        size (LongTensor): size of sets of shape :math:`(N,)`
        device (torch.device, optional): device of the tensor
    """
    rand = torch.rand(size.sum(), device=device)
    perm = variadic_sort(rand, size)[1]
    return perm


def one_hot(index, size):
    """
    Expand indexes into one-hot vectors.

    Parameters:
        index (Tensor): index
        size (int): size of the one-hot dimension
    """
    shape = list(index.shape) + [size]
    result = torch.zeros(shape, device=index.device)
    if index.numel():
        assert index.min() >= 0
        assert index.max() < size
        result.scatter_(-1, index.unsqueeze(-1), 1)
    return result


def clipped_policy_gradient_objective(policy, agent, reward, eps=0.2):
    ratio = (policy - agent.detach()).exp()
    ratio = ratio.clamp(-10, 10)
    objective = torch.min(ratio * reward, ratio.clamp(1 - eps, 1 + eps) * reward)
    return objective


def policy_gradient_objective(policy, reward):
    return policy * reward