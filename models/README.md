# Model registry

`registry.json` is the reviewed selection of model releases. Training creates
candidates; it never changes `active_version`. Current approval reflects the
user's visual preference for the larger model, not a claim of better validation
loss or universal superiority.

- `landscape-large-v1`: active, approved, 12.62M parameters, epoch 20.
- `landscape-small-v1`: previous approved model, 4.48M parameters, epoch 38.
- `landscape-clean-candidate-v1`: separate cleaner-data experiment, not approved.

The active weights are published at
https://huggingface.co/adamnouassida5/pixel-landscape-transformer
with a version tag and immutable commit recorded in the registry. Only the
active larger release has been published initially. Publish another approved
release before selecting it for a hosted rollback.

The source-checkpoint SHA identifies the original training checkpoint, while
`files` contains hashes of the smaller inference artifacts. These are different
files and their hashes need not match. The tokenizer and dataset manifest hashes
bind each release to its token vocabulary and training data. Validation losses
across different dataset manifests should not be compared directly.
