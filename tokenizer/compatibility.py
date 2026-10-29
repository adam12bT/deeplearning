"""Checks for decoder experiments that promise unchanged visual token meanings."""
import hashlib


def frozen_tokenizer_digest(model):
    result=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith(('encoder.','vq.')):
            result.update(name.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def assert_same_tokenizer(reference,candidate):
    for field in ('in_channels','base_channels','mid_channels','hidden_channels',
                  'embedding_dim','num_embeddings','beta','activation'):
        if getattr(reference.cfg,field)!=getattr(candidate.cfg,field):
            raise ValueError(f'Incompatible tokenizer configuration: {field}')
    if frozen_tokenizer_digest(reference)!=frozen_tokenizer_digest(candidate):
        raise ValueError('Encoder/codebook weights or normalization state differ; decoder override is unsafe')
