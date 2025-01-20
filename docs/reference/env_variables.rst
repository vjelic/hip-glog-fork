.. meta::
    :description: HIP environment variables reference
    :keywords: AMD, HIP, environment variables, environment, reference

********************************************************************************
HIP environment variables
********************************************************************************

In this section, the reader can find all the important HIP environment variables
on AMD platform, which are grouped by functionality.

GPU isolation variables
================================================================================

The GPU isolation environment variables in HIP are collected in the next table.
For more information, check :doc:`GPU isolation page <rocm:conceptual/gpu-isolation>`.

.. csv-to-list-table::
   :file: data/reference/env_variables.csv
   :rows: 12-14
   :columns: 0, 2
   :widths: 70,30


Profiling variables
================================================================================

The profiling environment variables in HIP are collected in the next table. For
more information, check :doc:`setting the number of CUs page <rocm:how-to/setting-cus>`.

.. csv-to-list-table::
   :file: data/reference/env_variables.csv
   :rows: 29-31
   :columns: 0, 2
   :widths: 70,30

Debug variables
================================================================================

The debugging environment variables in HIP are collected in the next table. For
more information, check :ref:`debugging_with_hip`.

.. csv-to-list-table::
   :file: data/reference/env_variables.csv
   :rows: 2-11

Memory management related variables
================================================================================

The memory management related environment variables in HIP are collected in the
next table.

.. csv-to-list-table::
   :file: data/reference/env_variables.csv
   :rows: 15-28
   :widths: 35,14,51

Other useful variables
================================================================================

The following table lists environment variables that are useful but relate to
different features.

.. csv-to-list-table::
   :file: data/reference/env_variables.csv
   :rows: 32
   :widths: 35,14,51
