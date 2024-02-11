Dependencies:
* liesl (https://pyliesl.readthedocs.io/en/latest/index.html)
* pylsl (https://pypi.org/project/pylsl/)

Note that you may need to install `liblsl` to run the project or else you may get errors like:

```
During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/ezrahuang/miniconda3/envs/ml/bin/liesl", line 5, in <module>
    from liesl.cli.main import main
  File "/home/ezrahuang/miniconda3/envs/ml/lib/python3.10/site-packages/liesl/__init__.py", line 1, in <module>
    from liesl.api import *
  File "/home/ezrahuang/miniconda3/envs/ml/lib/python3.10/site-packages/liesl/api.py", line 6, in <module>
    from liesl.streams.finder import open_stream, open_streaminfo
  File "/home/ezrahuang/miniconda3/envs/ml/lib/python3.10/site-packages/liesl/streams/finder.py", line 6, in <module>
    import pylsl
  File "/home/ezrahuang/miniconda3/envs/ml/lib/python3.10/site-packages/pylsl/__init__.py", line 2, in <module>
    from .pylsl import IRREGULAR_RATE, DEDUCED_TIMESTAMP, FOREVER, cf_float32,\
  File "/home/ezrahuang/miniconda3/envs/ml/lib/python3.10/site-packages/pylsl/pylsl.py", line 1335, in <module>
    raise RuntimeError(err_msg + "\n " + __dload_msg)
RuntimeError: liblsl library 'liblsl.so.2' found but could not be loaded - possible platform/architecture mismatch.

 You can install the LSL library with conda: `conda install -c conda-forge liblsl`
or otherwise download it from the liblsl releases page assets: https://github.com/sccn/liblsl/releases
```

files: 
* `test.py` - starts listening for LSL streams on the network
* `cmd` - command to create a dummy LSL stream

How to run:

in one window: `python test.py`
in another window copy the command from cmd and run: `liesl mock --type EEG`