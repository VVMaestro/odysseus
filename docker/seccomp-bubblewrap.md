# Native process containment in Docker

`seccomp-bubblewrap.json` is a locally modified Moby seccomp profile, not an
upstream profile. Its base is:

- Repository: https://github.com/moby/profiles
- Commit: `2ceae35d351c156cb5a8efc0fdc4a08cf94569d8`
- Source: `seccomp/default.json`
- License: Apache-2.0; see `seccomp-bubblewrap.LICENSE`.

All upstream rules remain. One additional allow rule permits `clone`,
`unshare`, `mount`, `umount2`, and `pivot_root`. Bubblewrap uses these calls
to create its user, mount, and PID namespaces. Linux capability checks still
apply. The profile does not add container capabilities or enable privileged
mode. Allowing namespace creation increases the kernel interfaces accessible
to processes in the container compared with the upstream default filter.

The Dockerfile installs bubblewrap. Compose supplies this profile to the
Docker daemon; it is not a file that the application loads. Invoke Compose
from the repository root so the profile's relative path resolves correctly.
After changing the profile, recreate the service; a container restart does
not update its seccomp policy. To apply a locally built image:

```sh
docker compose build odysseus
docker compose up -d --no-deps --no-build --pull never odysseus
```

Keep the source commit pinned when updating the profile. Compare upstream
rule changes separately from the local allow rule, and verify native process
execution and denial of access to application state before deploying it.
