# Virtual desktops

## Virtual desktop session drops after login
Check whether the virtual machine has enough memory and the host is not overloaded. Ask the user to sign out completely and sign in again. If a snapshot was recently taken, check that it did not leave the machine paused.

## Slow virtual machine
Look at CPU and memory use on the host. Remove old snapshots, because long snapshot chains slow the disk down.
