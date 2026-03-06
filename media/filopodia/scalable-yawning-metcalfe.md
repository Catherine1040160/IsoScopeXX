# Plan: Configure Claude Code StatusLine

## Task
Create the Claude Code settings file with statusLine configuration based on the user's shell PS1.

## Implementation

1. Create `/home/chang/.claude/settings.json` with the following content:

```json
{
  "statusLine": {
    "type": "command",
    "command": "debian_chroot=\"\"; [ -r /etc/debian_chroot ] && debian_chroot=$(cat /etc/debian_chroot); chroot=\"\"; [ -n \"$debian_chroot\" ] && chroot=\"($debian_chroot)\"; printf \"%s\\033[01;32m%s@%s\\033[00m:\\033[01;34m%s\\033[00m\" \"$chroot\" \"$(whoami)\" \"$(hostname -s)\" \"$(pwd)\""
  }
}
```

This will display: `user@host:current_directory` with green for user@host and blue for the directory path, matching the PS1 configuration.

## Files to Create
- `/home/chang/.claude/settings.json`
