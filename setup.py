from setuptools import setup

PKG_NAME = 'fgs_bayesian_filter'

setup(
    name=PKG_NAME,
    version='1.0.0',
    packages=[PKG_NAME],
    install_requires=['setuptools'],
    zip_safe=True,
    author='fugashy',
    author_email='fugashy@icloud.com',
    maintainer='fugashy',
    maintainer_email='fugashy@icloud.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: MIT :: MIT license',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description=('A package that provide bayasian filters such as kalman filter'),
    license='MIT',
    test_suite='test',
    entry_points={
        'console_scripts': [
            'start = fgs_bayesian_filter.ros_entry_point:main',
        ],
    },
)
