
Variational Learning in Neural Networks
=======================================

This page describes examples of how to use ensemble learning in neural
networks software (ENSMLP). The software includes variational
approximations that are Gaussian and mixtures of Gaussians used to
approximate neural network posterior distributions.

The ENSMLP software can be downloaded
[here](http://ml.sheffield.ac.uk/~neil/cgi-bin/software/downloadForm.cgi?toolbox=ensmlp).

Release Information
-------------------

Current release is 0.1.

As well as downloading the ENSMLP software you need to obtain the
toolboxes specified below. These can be downloaded using the *same*
password you get from registering for the ENSMLP software.

| **Toolbox**                                     |  **Version**  |
|-------------------------------------------------|---------------|
|  [NDLUTIL](/ndlutil/downloadFiles/vrs0p158)     |  0.158	  |
|  [NETLAB](/netlab/downloadFiles/vrs3p3)         |  3.3      	  |
|  [LIGHTSPEED](/lightspeed/downloadFiles/vrs2p1) |  2.1      	  |

First release in response to a request for the code. Code was written in
1998-1999 but is being released for first time in 2007. The code is
heavily based on the NETLAB toolbox, to such an extent that copyright
from NETLAB probably applies to large portions of this software. Please
see GPL licenses on that software for details of the implications of
this.

Examples
--------

### Tecator Data

The ensemble learning is demonstrated with a series of examples on the
['Tecator'](http://lib.stat.cmu.edu/datasets/tecator) data of Thodberg.

There are several different configurations of the models to run on the
Tecator data, all start with '`dem`'. I've put together this code
release over a couple of days, and haven't managed to recreate exactly
the results on this data we quote in our original tech report. Be aware
also that the scripts each take a few hours to run. Finally the
`demTecatorMixEns...` scripts can only be run once the corresponding
`demTecatorEns...` script has been run.

Finally there is a Gaussian process demo, `demTecatorGpRbfArd` that you
will need to download my [GP toolbox](../gp/) to run.

Page updated on Tue Dec 25 00:45:31 2007

|
[Disclaimer](http://www.manchester.ac.uk/aboutus/documents/disclaimer/ "Disclaimer")
|
[Privacy](http://www.manchester.ac.uk/aboutus/documents/privacy/ "Privacy")
| [Copyright
notice](http://www.manchester.ac.uk/aboutus/documents/copyright/ "Copyright Notice")
|
[Accessibility](http://www.manchester.ac.uk/aboutus/documents/accessibility/ "Accessibility")
| [Freedom of
information](http://www.manchester.ac.uk/aboutus/documents/foi/ "Freedom of information")
|
[Feedback](http://www.manchester.ac.uk/aboutus/contact/feedback/ "Feedback")
|

Please contact
[webmaster.cs@manchester.ac.uk](mailto:webmaster.cs@manchester.ac.uk)
with comments and suggestions
