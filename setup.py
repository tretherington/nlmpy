from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='nlmpy',
      version='1.0.0',
      description='A Python package to create neutral landscape models',
      long_description=readme(),
      author="Thomas R. Etherington, E. Penelope Holland, David O'Sullivan, Pierre Vigier",
      author_email='etheringtont@landcareresearch.co.nz',
      url='https://pypi.python.org/pypi/nlmpy',
      license='MIT',
      packages=['nlmpy'],
      install_requires=[
          'numpy',
          'scipy',
          'numba',
      ],
      zip_safe=False)
