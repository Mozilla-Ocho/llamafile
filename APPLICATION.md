# Application Notes

This Applicatio Notes is targeted at both model packagers and application developers and is for infomation that is not directly relevant to users who are just simply trying to use llamafiles in a standalone manner. Instead it is for developers who want their models to better integrate with other developers models or applications. (Think of this as an informal ad-hoc community standards page)

## Finding Llamafiles

While we do not have a package manager for llamafiles, applications developers are recommended
to search for AI models tagged as `llamafile` in Hugging Face AI repository.
Be sure to display the publishing user or organisation and to sort by heart count.

Within a llamafile repository entry in Hugging Face, there may be multiple `*.llamafile` files
to choose from. The current convention to describe each sub entries of llamafiles is to 
insert a table in the model card surrounded by html comment start and end marker named `*-provided-files`.

For example a model card for a llamafile should have this section that you can parse:

```markdown
<!-- README_llamafile.md-provided-files start -->
## Provided files

| Name | Quant method | Bits | Size | Max RAM required | Use case |
| ---- | ---- | ---- | ---- | ---- | ----- |
| [phi-2.Q2_K.llamafile](https://huggingface.co/jartine/phi-2-llamafile/blob/main/phi-2.Q2_K.llamafile) | Q2_K | 2 | 1.17 GB| 3.67 GB | smallest, significant quality loss - not recommended for most purposes |
... further llamafile entries here ...

<!-- README_llamafile.md-provided-files end -->
```

## Packaging A Llamafile

The file naming convention for each llamafile is `<model-name>.<Quantisation Level>.llamafile` (e.g. `phi-2.Q2_K.llamafile`).

## Installing A Llamafile

Llamafile are completely standalone and portable so does not require installation.
However we have a path convention for ease of discovery by local applications scripts/program.

- **Linux** : `~/.llamafile/*.llamafile`

<!-- TODO: Windows llamafile installation convention -->

<!-- TODO: Mac llamafile installation convention -->
