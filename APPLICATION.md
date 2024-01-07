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

Llamafiles follow a naming convention of `<Model>-<Version>-<ExpertsCount>x<Parameters>-<Quantization>.llamafile`.

The components are:
1. **Model**: A descriptive name for the model type or architecture.
2. **Version (Optional)**: Denotes the model version number, starting at `v1` if not specified, formatted as `v<Major>.<Minor>`.
    - Best practice to include model version number only if model has multiple versions and assume the unversioned model to be the first version and/or check the model card.
3. **ExpertsCount**: Indicates the number of experts found in a Mixture of Experts based model.
4. **Parameters**: Indicates the number of parameters and their scale, represented as `<count><scale-prefix>`:
    - `T`: Trillion parameters.
    - `B`: Billion parameters.
    - `M`: Million parameters.
    - `K`: Thousand parameters.
5. **Quantization**: This part specifies how the model parameters are quantized or compressed. The notation is influenced by the `./quantize --help` command in `llama.cpp`.
   - Uncompressed formats:
     - `F16`: 16-bit floats per weight
     - `F32`: 32-bit floats per weight
   - Quantization (Compression) formats:
     - `Q<X>`: X bits per weight, where `X` could be `4` (for 4 bits) or `8` (for 8 bits) etc...
     - Variants provide further details on how the quantized weights are interpreted:
       - `_K`: k-quant models, which further have specifiers like `_S`, `_M`, and `_L` for small, medium, and large, respectively, if they are not specified, it defaults to medium.
       - `_<num>`: Different approaches, with even numbers indicating the model weights as a scaling factor multiplied by the quantized weight and odd numbers indicating the model weights as a combination of an offset factor plus a scaling factor multiplied by the quantized weight. This convention was found from this [llama.cpp issue ticket on QX_4](https://github.com/ggerganov/llama.cpp/issues/1240).
            - Even Number (0 or 2): `<model weights> = <scaling factor> * <quantised weight>`
            - Odd Number (1 or 3): `<model weights> = <offset factor> + <scaling factor> * <quantised weight>`


## Installing A Llamafile And Making It Accessible To Other Local Applications

Llamafiles are designed to be standalone and portable, eliminating the need for a traditional installation. For optimal discovery and integration with local application scripts/programs, we recommend the following search paths:

- **System-wide Paths**:
    - `/usr/share/llamafile` (Linux/MacOS/BSD): Ideal for developers creating packages, commonly accessed via package managers like `apt get install` in Debian-based Linux OSes.
    - `/opt/llamafile` (Linux/MacOS/BSD): Positioned in the `/opt` directory, suitable for installers downloaded directly from the web.
    - `C:\llamafile` (Windows): A direct path for Windows systems.

- **User-specific Path**:
    - `~/.llamafile` (Linux/MacOS/BSD): Located in the user's home directory, facilitating user-specific configurations in line with Unix-like conventions.

For applications or scripts referencing the Llamafile path, setting the environment variable `$LLAMAFILE_PATH` to a singular path can enhance configuration simplicity and system consistency.
