==========================
 Welcome to repo_template
==========================

|Build Status| |License| |codecov| |GitHub|


*Some general info*

https://github.com/anwarnunez/repo_template


What is repo_template?
======================

*A long description of the repo.*

The directory `gendoc` contains scripts neceesary for building website using sphinx and sphinx gallery.
The `.travis.yml` file contains the commands needed for travis-ci to deploy the website to GitHub pages.

The resulting template website can be seen here: https://anwarnunez.github.io/repo_template/


Installation
============

Clone the repo from GitHub and do the usual python install

.. code-block:: bash

   git clone https://github.com/anwarnunez/repo_template.git
   cd repo_template
   sudo python setup.py install


Getting started
===============

Requires minor configuration

Create GitHub access token
--------------------------

This enables travis to push and create the gh-pages branch.

github.com -> settings ("public profile") -> developer settings -> personal access tokens -> generate new token

* Under **Note** write something descriptive (e.g. `myrepo-travis-ghpages`).
* Under  **Select scopes** select **repo**.
* Finally, click on **Generate token** and copy the token.

Add token to travis repo settings
---------------------------------

travis-ci -> dashboard -> select repo -> settings

* Under **Environmental variables**, fill out:
  * *NAME*: `GITHUB_TOKEN`
  * *VALUE*: [paste the token]
* Finally, click **ADD**



.. Badge shortcuts:

.. |Build Status| image:: https://travis-ci.com/anwarnunez/repo_template.svg?branch=master
   :target: https://travis-ci.com/anwarnunez/repo_template

.. |License| image:: https://img.shields.io/badge/license-BSD%203--Clause-blue
   :target: https://opensource.org/licenses/BSD-3-Clause

.. |codecov| image:: https://codecov.io/gh/anwarnunez/repo_template/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/anwarnunez/repo_template

.. |GitHub| image:: https://img.shields.io/badge/github-repo_template-blue
   :target: https://github.com/anwarnunez/repo_template
