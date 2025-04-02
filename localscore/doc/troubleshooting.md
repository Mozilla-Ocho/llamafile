# Troubleshooting the LocalScore CLI

Much of the troubleshooting for LocalScore is similar to Llamafile.

## CLI

### Windows

On Windows, there's a variety of issues.

#### Model Size

If you are using a Llamafile or LocalScore benchmark bundle larger than 4 gigabytes, you will not be able to run it on Windows due to Windows limitations. You can use LocalScore as a standalone utility and pass in models in GGUF format.

#### WSL2

On WSL, there are many possible gotchas. One thing that helps solve them
completely is this:

```
[Unit]
Description=cosmopolitan APE binfmt service
After=wsl-binfmt.service

[Service]
Type=oneshot
ExecStart=/bin/sh -c "echo ':APE:M::MZqFpD::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"

[Install]
WantedBy=multi-user.target
```

Put that in `/etc/systemd/system/cosmo-binfmt.service`.

Ensure that the APE loader is installed to `/usr/bin/ape`:

```sh
sudo wget -O /usr/bin/ape https://cosmo.zip/pub/cosmos/bin/ape-$(uname -m).elf
sudo chmod +x /usr/bin/ape
```

Then run `sudo systemctl enable --now cosmo-binfmt`.

Another thing that's helped WSL users who experience issues, is to
disable the WIN32 interop feature:

```sh
sudo sh -c "echo -1 > /proc/sys/fs/binfmt_misc/WSLInterop"
```

In Windows 11 with WSL 2 the location of the interop flag has changed, as such
the following command be required instead/additionally:

```sh
sudo sh -c "echo -1 > /proc/sys/fs/binfmt_misc/WSLInterop-late"
```

In the instance of getting a `Permission Denied` on disabling interop
through CLI, it can be permanently disabled by adding the following in
`/etc/wsl.conf`

```sh
[interop]
enabled=false
```

#### Other Issues

We have observed that on Windows the performance of LocalScore is slower than on Linux. This is expected at the moment.

### Linux 

On some Linux systems, you might get errors relating to `run-detectors`
or WINE. This is due to `binfmt_misc` registrations. You can fix that by
adding an additional registration for the APE file format llamafile
uses:

```sh
sudo wget -O /usr/bin/ape https://cosmo.zip/pub/cosmos/bin/ape-$(uname -m).elf
sudo chmod +x /usr/bin/ape
sudo sh -c "echo ':APE:M::MZqFpD::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"
sudo sh -c "echo ':APE-jart:M::jartsr::/usr/bin/ape:' >/proc/sys/fs/binfmt_misc/register"
```

### Mac

On macOS with Apple Silicon you need to have Xcode Command Line Tools
installed for llamafile to be able to bootstrap itself.

If you use zsh and have trouble running llamafile, try saying `sh -c
./llamafile`. This is due to a bug that was fixed in zsh 5.9+. The same
is the case for Python `subprocess`, old versions of Fish, etc.


#### Mac error "... cannot be opened because the developer cannot be verified"

1. Immediately launch System Settings, then go to Privacy & Security. llamafile should be listed at the bottom, with a button to Allow.
2. If not, then change your command in the Terminal to be `sudo spctl --master-disable; [llama launch command]; sudo spctl --master-enable`. This is because `--master-disable` disables _all_ checking, so you need to turn it back on after quitting llama. 