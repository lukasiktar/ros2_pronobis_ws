from setuptools import setup, find_packages

package_name = 'microsegnet_inference'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name, package_name + '.*']),  # Ensure subpackages are included
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS2 package using modified TransUNet',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'microsegnet_inference = microsegnet_inference.microsegnet_inference:main',
        ],
    },
)
