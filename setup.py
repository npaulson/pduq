from os import path
from setuptools import setup, find_packages
import sys
import versioneer


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 6)
if sys.version_info < min_version:
    error = """
pduq does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*(sys.version_info[:2] + min_version))
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()


setup(
    name='pduq',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python package for uncertainty quantification in CALPHAD",
    long_description=open('README.rst').read(),  # Point to README.rst
    long_description_content_type='text/x-rst',  # Specify ReStructuredText content type
    author="Argonne National Laboratory",
    author_email='tguannan@anl.gov',
    url='https://github.com/npaulson/pduq/tree/GT_fix_ver',
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)) + ',<3.13',
    packages=find_packages(exclude=['docs', 'tests']),
    entry_points={
        'console_scripts': [
            # 'command = some.module:some_function',
            ],
        },
    include_package_data=True,
    package_data={
        'pduq': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
            'pduq/tests/trace.csv',
            'pduq/tests/CU-MG_param_gen.tdb'
            ]
        },
    install_requires=[
        'dask>=2',
        'pycalphad>=0.11.0',
        'numpy>=1.20',
        'scipy',
        'seaborn',
    ],
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
