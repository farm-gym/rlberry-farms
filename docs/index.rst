.. _rlberry-farms: https://gitlab.inria.fr/scool/rlberry-farms

.. _index:

Farming with RL agents
======================

rlberry-farms is a collection of Reinforcement-Learning `gym <https://github.com/openai/gym>`_  environments featuring agricultural games constructed via `farm-gym <https://gitlab.inria.fr/rl4ae/farm-gym>`_. These environments are toy environments and they are not meant to represent exactly the real-life challenges of a farmer but they are meant to illustrate some of the problems that a scientist would encounter when deploying a RL agent in the field.



.. _installation:

Installation
============

To install this repo, one can use pip. It is advised to use a virtual environment in order to avoid conflicting library (see `python website  <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment>`_ ).

.. code:: bash

    $ pip install git+https://gitlab.inria.fr/scool/rlberry-farms


To confirm the installation, you can run

.. code:: bash

    $ python examples/installation_test.py


To try the environments manually (to be the agent), you can use :

.. code:: bash

    $ python examples/interactive_farm0.py
    $ python examples/interactive_farm1.py
    

Documentation Contents
======================

We provide two small guides: how to make an expert agent and how to use ppo in :class:`rlberry_farms.Farm1`.

.. toctree::
   :maxdepth: 2

   tuto/Tuto_AgentExpert.rst
   tuto/tuto_ppo.rst

See also the :ref:`the examples<examples>` and the :ref:`API<api>` sections.
