# New Backend

This new backend relies and a shared memory buffer to separate 
capturing and storing an image sequence from a model's implementation.

### usage 
`python test_newbackend/app.py` 

At the moment no real user interface.
This demo uses opencv windows to display chart and most recent image frame.

### TODOs
- [ ] integrate with GUI
- [ ] clean up and more explaination

**Notes:**
- 2 files in this folder are copied from `./src/app/`
  - DemoModel.py, demo.ckp
- RingBuffer implementation: see: https://github.com/ctrl-labs/cringbuffer
  - Changes have been made to allow the writer to overwrite entries before they are read by the model class.
    This allows for readers to always have the newest frame.
