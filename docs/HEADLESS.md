# Prepare a clean headless DGX Spark

Use this before building or loading the model on a dedicated DGX Spark / GB10.
The aim is a running Linux system with networking, SSH and required services,
without a desktop or another large inference engine consuming unified memory.
This does not reinstall the OS or erase models, environments or caches.

## 1. Keep a working remote shell

Save desktop work first. Switching targets ends the graphical session and can
stop services that are not part of the destination target. Run the change from
SSH or a physical text console, not a terminal inside the desktop.

Record the current boot target and check SSH on the DGX OS Ubuntu installation:

```bash
systemctl get-default
systemctl status ssh.service --no-pager
```

If SSH is installed but inactive, enable it before proceeding:

```bash
sudo systemctl enable --now ssh.service
```

Open a second SSH connection from another device and verify it works. If using
Tailscale, check `tailscale status` and `systemctl status tailscaled.service` too.
Keep SSH, networking and the remote-access service running throughout.

## 2. Stop competing inference workloads

Inspect services, processes and listeners before stopping anything:

```bash
systemctl list-units --type=service --state=running --no-pager
systemctl --user list-units --type=service --state=running --no-pager
ps -eo pid,ppid,comm,args --sort=pid
sudo ss -ltnp
nvitop --once --no-unicode
```

If Docker or tmux is installed, inspect its managed workloads as well:

```bash
docker ps --format 'table {{.ID}}	{{.Names}}	{{.Status}}'
tmux list-sessions
```

Stop each identified inference engine through the manager that launched it.
For a foreground `scripts/serve.sh`, use Ctrl+C in that server's terminal and
wait for its workers to exit. For managed services or containers, substitute
the exact name found above in the appropriate command:

```bash
sudo systemctl stop YOUR_INFERENCE_UNIT.service
systemctl --user stop YOUR_INFERENCE_UNIT.service
docker stop YOUR_INFERENCE_CONTAINER
```

These are alternatives for different launch methods. Do not stop unrelated
services or every Python process. A stopped container or service may have a
separate supervisor that restarts it; verify the process has actually exited.
If preparing a reboot, inspect that workload's autostart policy too. For a
systemd inference unit that should stay off across boots, use
`sudo systemctl disable --now YOUR_INFERENCE_UNIT.service`; record it so you
can restore it later. Keep Docker storage and model files intact.

## 3. Switch to headless multi-user mode

For the current boot only:

```bash
sudo systemctl isolate multi-user.target
```

To also boot headless from now on:

```bash
sudo systemctl set-default multi-user.target
```

`isolate` changes the running system. `set-default` changes future boots.
No reboot is required for the target switch. Multi-user mode still runs enabled
system services, so it does not by itself unload model servers or containers.

Verify the result from SSH:

```bash
systemctl get-default
systemctl is-active multi-user.target
systemctl is-active graphical.target
systemctl is-active display-manager.service
```

Expect multi-user to be active and the graphical target/display manager to be
inactive. An absent display-manager unit is normal on a server-only install.
`is-active` returns a nonzero exit status for inactive or absent units. If you
only used `isolate`, `get-default` can still report the original boot target.

## 4. Verify the machine is ready

```bash
nvitop --once --no-unicode
free -h
vmstat 1 5
sudo ss -ltnp
```

There should be no competing inference compute process and no listener on the
port selected for the new server, normally 8092. Use the process list and port
owner together; an empty port alone does not prove every worker has exited.
On this GB10 workflow, use nvitop and system memory counters for telemetry.

Read `available` memory in `free`, not only `free`. Linux uses spare RAM for
reclaimable file cache. Swap occupancy alone does not prove current thrashing;
look at the later `vmstat` samples for ongoing swap-in/swap-out (`si`/`so`).
There is no universal idle-memory threshold across installations. The qualified
model used about 117.75 GiB of 121.62 GiB total after testing, so leave the
machine dedicated to this one large engine and avoid building alongside it.

Preserve the FlashInfer/compiler caches and the checkpoint's mmap-backed PLE
pages. Routine `drop_caches` or `swapoff -a` is not part of this procedure.
Dropping caches makes the next load/prefill colder, and disabling swap under
memory pressure can fail. The kernel already reclaims clean cache as needed.

For a fresh boot baseline, after saving work and handling inference autostart:

```bash
sudo reboot
```

Reconnect and repeat the checks above. A reboot is optional; it is not a
substitute for identifying services that automatically reload models.

## 5. Start one engine

From the cloned repository, after completing the build and download steps:

```bash
export QWEN_SETUP_ROOT="$HOME/qwen38-spark"
bash scripts/serve.sh
```

Keep it in the foreground or launch it inside a dedicated tmux session. In a
second shell, check readiness and the exact served model:

```bash
curl -fsS http://127.0.0.1:8092/health
curl -fsS http://127.0.0.1:8092/v1/models
```

Continue with [setup and prefix-cache validation](SETUP.md#4-check-an-answer-and-prefix-reuse).

## Restore the desktop

Stop the large model first so the desktop has memory available. To bring the
desktop back now and make graphical boot the default again:

```bash
sudo systemctl set-default graphical.target
sudo systemctl isolate graphical.target
```

This assumes a desktop and display manager remain installed. If your original
boot target was different, restore the target recorded in step 1 instead.
Re-enable only the inference units you deliberately disabled and want back.

## Official references

- [systemd systemctl reference](https://www.freedesktop.org/software/systemd/man/latest/systemctl.html): isolate and set-default semantics.
- [systemd target definitions](https://www.freedesktop.org/software/systemd/man/latest/systemd.special.html): multi-user and graphical targets.
- [Linux kernel VM documentation](https://www.kernel.org/doc/html/latest/admin-guide/sysctl/vm.html#drop-caches): cache reclaim and the cost of dropping caches.
