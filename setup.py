import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'oddsmaker'
AUTHOR = 'Blake Atkinson'
AUTHOR_EMAIL = 'blaketatkinson@gmail.com'
URL = 'https://github.com/btatkinson/moneyball'

LICENSE = 'MIT License'
DESCRIPTION = 'Speedy Implementation of Team and Player Rating Algorithms'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'pandas'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )