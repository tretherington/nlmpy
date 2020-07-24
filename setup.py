from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='nlmpy',
      version='0.1.5',
      description='A Python package to create neutral landscape models',
      long_description=readme(),
      author="Thomas R. Etherington, E. Penelope Holland, and David O'Sullivan",
      author_email='t.etherington@kew.org',
      url='https://pypi.python.org/pypi/nlmpy',
      license='MIT',
      packages=['nlmpy'],
      install_requires=[
          'numpy',
          'scipy',
      ],
      zip_safe=False)
