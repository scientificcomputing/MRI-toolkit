# Info command

The `info` command allows you to quickly inspect the metadata of an MRI file. It displays the image shape, voxel size, data type, and the affine transformation matrix.

## Usage

```bash
mritk info <file_path> [OPTIONS]
```

**Arguments:**
* `file`: Path to the file to display information about.

**Options:**
* `--json`: Output information in JSON format. Useful for programmatic parsing.


![info](https://github.com/user-attachments/assets/fc0e734d-3c94-48fa-8e25-3e65bfc86ebe)
