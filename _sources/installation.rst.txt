Installation
============

Clone the repository and install the package in editable mode.

Command-line steps
-------------------

1. Clone the repository (replace the URL with your remote if different):

.. code-block:: bash

	git clone https://github.com/quentin-duchemin/DisTreebution.git
	cd DisTreebution

2. Create and activate a virtual environment (recommended):

.. code-block:: bash

	python -m venv .venv
	source .venv/bin/activate

3. Install the package (editable) and runtime dependencies:

.. code-block:: bash

	pip install -e .
	pip install numpy matplotlib pandas

Notes
-----
- Editable install (`pip install -e .`) lets you modify the code locally while using the package from any Python process in the environment.

- The quick-start example `package_docs/examples/quick_start.py` adjusts `sys.path` so it can be run from the repository root without installing; however, installing the package makes imports cleaner.

