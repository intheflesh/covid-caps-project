# covid-caps-project
a capsule network based project to detect COVID19 in lung X-Rays
This project belongs to Noga Mudrik and Roman Frenkel.

This project is based on the covid-caps project:
https://github.com/ShahinSHH/COVID-CAPS

We did several changes to it:
* pre-processed the input images to remove noises such as letters, markings, numbers, etc.
* added code to handle missing files we found in the pre-training dataset.
* changed the exporting of images to numpy arrays to be in batches, so the memory does not overflow and the runtime is reduced.


