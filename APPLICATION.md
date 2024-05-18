# Application Notes

This Application Notes is targeted at both model packagers and application developers and is for information that is not directly relevant to users who are just simply trying to use llamafiles in a standalone manner. Instead it is for developers who want their models to better integrate with other developers models or applications. (Think of this as an informal ad-hoc community standards page)

## Finding Llamafiles

While we do not have a package manager for llamafiles, applications developers are recommended
to search for AI models tagged as `llamafile` in Hugging Face AI repository.
Be sure to display the publishing user or organisation and to sort by trending.

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

## Llamafile Naming Convention

Llamafiles follows the same naming convention as gguf but instead of `.gguf` its `.llamafile`. Consult [gguf naming convention]([https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#gguf-naming-convention)) for specifics.


## Installing A Llamafile And Making It Accessible To Other Local Applications

Llamafiles are designed to be standalone and portable, eliminating the need for a traditional installation. For optimal discovery and integration with local application scripts/programs, we recommend the following search paths:

- **System-wide Paths**:
    - `/usr/share/llamafile` (Linux/MacOS/BSD): Ideal for developers creating packages, commonly accessed via package managers like `apt get install` in Debian-based Linux OSes.
    - `/opt/llamafile` (Linux/MacOS/BSD): Positioned in the `/opt` directory, suitable for installers downloaded directly from the web.
    - `C:\llamafile` (Windows): A direct path for Windows systems.

- **User-specific Path**:
    - `~/.llamafile` (Linux/MacOS/BSD): Located in the user's home directory, facilitating user-specific configurations in line with Unix-like conventions.

For applications or scripts referencing the Llamafile path, setting the environment variable `$LLAMAFILE_PATH` to a singular path can enhance configuration simplicity and system consistency.
