from torch.autograd import Variable
import torch
import numpy as np


def hamming_distance_sample(sents: torch.Tensor, tau: int, bos_id: int, eos_id: int, pad_id: int, vocab_size: int) -> torch.Tensor:
    """
    Sample a batch of corrupted examples from sents.
    Args:
        sents: Tensor [batch_size, n_steps]. The input sentences.
        tau: Temperature (int) (0 < tau < 1).
        vocab_size: to create valid samples (int).
        bos_id: id of the beginning of sentence token (int).
        eos_id: id of the end of sentence token (int).
        pad_id: id of the padding token (int).
    Returns:
        sampled_sents: Tensor [batch_size, n_steps]. The corrupted sentences.
    """

    mask = torch.eq(sents, bos_id) | torch.eq(
        sents, eos_id) | torch.eq(sents, pad_id)
    lengths = mask.mul(-1).add(1).float().sum(dim=1)
    batch_size, n_steps = sents.size()

    # first, sample the number of words to corrupt for each sentence
    logits = torch.arange(n_steps)
    logits = logits.mul_(-1).unsqueeze(0).expand_as(
        sents).contiguous().masked_fill_(mask, np.iinfo(np.int64).min)
    logits = Variable(logits)
    probs = torch.nn.functional.softmax(logits.float().mul_(tau), dim=1)
    num_words = torch.distributions.Categorical(probs).sample()

    # sample the corrupted positions.
    corrupt_pos = num_words.data.float().div(lengths).unsqueeze(
        1).expand_as(sents).contiguous().masked_fill_(mask, 0)
    corrupt_pos = torch.bernoulli(corrupt_pos, out=corrupt_pos).bool()
    total_words = int(corrupt_pos.sum())

    # sample the corrupted values, which will be added to sents
    corrupt_val = torch.LongTensor(total_words)
    corrupt_val = corrupt_val.random_(1, vocab_size)
    corrupts = torch.zeros(batch_size, n_steps).long()
    corrupts = corrupts.masked_scatter(corrupt_pos, corrupt_val)
    sampled_sents = sents.add(Variable(corrupts)).remainder(vocab_size)
    return sampled_sents
