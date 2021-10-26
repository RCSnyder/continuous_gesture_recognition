# New Backend
This new backend relies and a shared memory buffer to separate the
capturing and storing of image sequences from a model implementation.

### usage 
`python test_newbackend/main.py` 
> tested on: debian bullseye, python v3.9.2, ROCm stack v4.3.0

At the moment no real user interface.
This demo uses opencv windows to display chart and most recent image frame.
Runs for 2 minutes then kills self, `ctrl+c` should kill early.

### TODOs
- [ ] integrate with GUI
  - [ ] `@ianzur`: expected it to be possible to use with a flask backend similar to [celery](), but did not investigate implementing.
    > for a web app it may make more sense to move towards a java implementation
- [ ] instead of hacking in the changes into ringbuffer, subclass

**Notes:**
- 2 files in this folder are directly copied from `./src/app/`
  - model structure definition: `DemoModel.py`
  - model weights: `demo.ckp`
- RingBuffer implementation: see: https://github.com/ctrl-labs/cringbuffer
  - Changes:
    - writer allowed to overwrite entries before they are read by the model class.
      > This allows for readers to always have the newest frame. (in the case of slow model execution, camera fps remains constant)
    - reader pointers ignored, does not track where the readers are. Reader always reads `n`-most recent frames. Writer position is used to locate the most recent frame.
  
**contact**
- questions, concerns? raise an issue and `@ianzur` or send me an email `ian dot zurutuza at gmail dot com`
