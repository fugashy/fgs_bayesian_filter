from setuptools import setup

PKG_NAME = 'fgs_gaussian_filter'

setup(
    name=PKG_NAME,
    version='1.0.0',
    packages=[PKG_NAME],
    data_files=[('share/' + PKG_NAME, ['config/gaussian_filter.yaml', 'config/application.yaml'])],
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
    description=('A package that provide gaussian filters such as kalman filter'),
    license='MIT',
    test_suite='test',
    entry_points={
        'console_scripts': [
            'filter_out = fgs_gaussian_filter.ros2_entry_point:ekf_sample',
        ],
    },
)
