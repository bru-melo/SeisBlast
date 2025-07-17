# setup.py

from setuptools import setup, find_packages

setup(
    name='SeisBlast',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib'
    ],
    #entry_points={
    #    'console_scripts': [
    #        'list-picks=seisblast.compare_picktimes:list_picks',
    #        'diff-picks=seisblast.compare_picktimes:diff_picks',
    #        'get_wave=seisblast.compare_picktimes:get_wave',
    #        'view_wave=seisblast.compare_picktimes:view_wave'
    #    ],
    #},
    #url='https://github.com/yourusername/SeisBlast',
    #license='MIT',
    #author='Bruna Melo',
    #author_email='bmelo@cp.dias.ie',
    description='A package for reading, comparing and plotting quarry blast seismic data'
)

