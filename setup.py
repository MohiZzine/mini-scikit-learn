from setuptools import setup, find_packages

setup(
    name='Mini Scikit-Learn Library',
    version='0.1.0',
    author='Mohieddine Farid | Image Fjer',
    author_email='farid.mohieddine@um6p.ma',
    description='A lightweight machine-learning library inspired by Scikit-Learn',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MohiZzine/mini-scikit-learn',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.5',
        'pandas>=1.0.5',
        'scikit-learn>=0.23.1' 
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)

