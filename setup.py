from setuptools import setup, find_packages

setup(
    name='optimized_dw_nca',
    version='0.1.0',
    author='UBANDIYA Najib Yusuf',
    author_email='najibubandia@gmail.com',
    description='Optimized implementation of Distance-Weighted Neighborhood Components Analysis (DW-NCA)',
    long_description=open('README.md').read(),  # Assumes you have a README.md file
    long_description_content_type='text/markdown',
    url='https://github.com/ubandiya/optimized_dw_nca',  # Replace with your GitHub repo URL
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.6.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.4.0',  # Include other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    test_suite='tests',  # Indicates the directory for test discovery
    tests_require=[
        'unittest',
    ],
)
