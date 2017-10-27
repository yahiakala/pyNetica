==================================================
pyNetica (netica): Python wrapper for Netica C API
==================================================

Prerequisites
-------------
This package requires the Netica C API Library (.dll or .so) to be inserted
into the /lib folder. So far, it has only been tested on Windows (.dll).
You must obtain the library yourself
from www.norsys.com. There is a free version for download which allows
you to manipulate networks of a limited size. Note that Netica offers two
mutually exclusive (and similarly priced) licenses for users who want to
work with large networks:

* Netica Application license (Work through the GUI only)
* Netica API Family license (C, C#, C++, Java, Matlab, Visual Basic, etc.)

Check out http://www.norsys.com/onLineAPIManual/index.html
for help on the C API functions.

Installation
------------
* Fork the repo (or just clone if you don't want to develop)
* Clone to your desktop
* Navigate to the folder and in your prompt:

.. code:: bash

    pip install .
    # or
    pip install -e .  # developer install.

Comparison to SVN Repository Package
-------------------------------------------------------
Many simplifications to the workflow were made for typical users working
in single environments and single nets.

Old workflow:

.. code:: bash

    import netica
    ntc = netica.Netica()
    env = ntc.newenv()
    ntc.initenv()
    net_p = ntc.newnet('BayesNet', env)
    # or
    net_p = ntc.opennet(env, 'file.dne')

    # Do your processing with redundant input args from above

New workflow:

.. code:: bash

    import netica
    ntc = netica.NeticaNetwork()
    # or
    ntc = netica.NeticaNetwork(openfile='file.dne')

    # Do your processing entirely with methods on ntc, no redundancy

Credits
-------
Credit goes to Kees den Heijer (c.denheijer@tudelft.nl) for getting this started.
