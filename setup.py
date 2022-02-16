import setuptools

setuptools.setup(
    name="apomock",
    version="0.1",
    author="James Lane",
    author_email="lane@astro.utoronto.ca",
    description="Mock data for APOGEE",
    packages=setuptools.find_packages(include=['apomock','apomock.util']),
    url='http://github.com/jamesmlane/apomock',
    install_requires=['numpy','scipy','galpy','healpy','astropy']
)