
# vo-box-lite
Video odometry on an extremely resource constrained ESP32-S3.

## REPO IDIOMS ALWAYS FOLLOW
-
- default to inaction: if a task runs into an unexpected speedbump or is ambiguous or requires you to make a big decision, stop and ask for input from the user
- default to small deltas: this repo is still a WIP and a lot of code fully doesn't exist yet - prefer small incremental changes even if they leave a feature unimplemented
- update this agents.md file as you go, but keep edits as concise as possible - should only be for commands, architectural changes, etc that will need to be used by future agents

## Algorithm
1. Offline
- The S3 module takes a video of the environment
- It is streamed back to the user's laptop over WiFi
- The laptop processes the features extracted from the video using COLMAP into a 3d feature map
- The feature map is streamed back to the S3

2. Online
- A frame is capture
- It is downscaled to 1x size, 1.2x size, 1.2*1.2x size, etc (7 total sizes)
- 5x5 box blur
- FAST12
- rBRIEF
- mutually nearest neighbour + Lowe's ratio test feature matching
- PnP RANSAC

## Architecture
### ESP32-S3
Rust using the esp-hal crate, largely unimplemented right now.

### Laptop
Not implemented right now.


## Useful Commands
- `. /home/north/export-esp.sh` - loads the esp toolchain into path

